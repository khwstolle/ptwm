{
  description = "Format for PyTorch module weights.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";

    # Formatting tools.
    treefmt.url = "github:numtide/treefmt-nix";

    # Nightly Rust toolchains.
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    # Pure-Nix Python packaging from pyproject.toml + uv.lock.
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs @ {flake-parts, ...}:
    flake-parts.lib.mkFlake {inherit inputs;} {
      imports = [
        inputs.treefmt.flakeModule
      ];
      systems = [
        "x86_64-linux"
      ];
      perSystem = {system, ...}: let
        pkgs = import inputs.nixpkgs {inherit system;};
        lib = pkgs.lib;
        fenixPkgs = inputs.fenix.packages.${system};

        # Libraries that prebuilt PyPI wheels (torch, accelerate, …) open at
        # runtime. uv2nix-installed wheels are autopatched where possible,
        # but a few still need runtime LD lookups for libstdc++ / libGL /
        # CUDA stubs.
        systemLibs = with pkgs; [
          stdenv.cc.cc.lib # libstdc++.so.6
          zlib # libz.so
          glib # libglib-2.0.so
          libxkbcommon
          libGL
        ];
        libPath = lib.makeLibraryPath systemLibs;

        # ── Python environment via uv2nix ──────────────────────────────
        python = pkgs.python313;

        workspace = inputs.uv2nix.lib.workspace.loadWorkspace {
          workspaceRoot = ./.;
        };

        # Overlay that turns the uv-resolved set into Nix packages. We
        # prefer wheels — torch from source would be untenable.
        uvOverlay = workspace.mkPyprojectOverlay {
          sourcePreference = "wheel";
        };

        # Local-package overrides. ptwm itself is built by maturin against
        # the Rust workspace; uv2nix's default scaffold uses
        # `setuptools` so we re-wire it through `cargoSetupHook`. The
        # `pyproject-build-systems` overlay supplies the maturin build
        # hook itself.
        # Wheels that pull CUDA siblings (or each other) at runtime — every
        # one of these needs auto-patchelf relaxed because the referenced
        # `.so` lives in a sibling wheel under a separate Nix store path.
        # Runtime resolution still works: `torch.__init__` rewrites
        # `LD_LIBRARY_PATH` on import, so the venv finds the libs through
        # the per-wheel install paths.
        crossWheelLinkingPackages = [
          "torch"
          "torchvision"
          "torchaudio"
          "triton"
          "xformers"
          "bitsandbytes"
        ];

        pyprojectOverrides = _final: prev:
          (lib.mapAttrs (
              name: pkg:
                if lib.hasPrefix "nvidia-" name || lib.elem name crossWheelLinkingPackages
                then
                  pkg.overrideAttrs (_old: {
                    autoPatchelfIgnoreMissingDeps = true;
                  })
                else pkg
            )
            prev)
          // {
            ptwm = prev.ptwm.overrideAttrs (old: {
              cargoDeps = pkgs.rustPlatform.importCargoLock {
                lockFile = ./Cargo.lock;
              };
              nativeBuildInputs =
                (old.nativeBuildInputs or [])
                ++ [
                  pkgs.rustPlatform.cargoSetupHook
                  pkgs.rustc
                  pkgs.cargo
                  # `git2` with `vendored-openssl` pulls openssl-src, which
                  # invokes `perl ./Configure` during the build. Without
                  # perl in the sandbox, openssl-build fails before maturin
                  # finishes compiling the extension.
                  pkgs.perl
                ];
            });
          };

        pythonSet =
          (pkgs.callPackage inputs.pyproject-nix.build.packages {
            inherit python;
          }).overrideScope
          (
            lib.composeManyExtensions [
              inputs.pyproject-build-systems.overlays.default
              uvOverlay
              pyprojectOverrides
            ]
          );

        # Resolve dependency groups against the lockfile. `workspace.deps.all`
        # picks up every optional + dev group declared in `pyproject.toml`
        # ([dev, integration, docs]) — fine for a single Nix-built venv that
        # serves both the test loop and the docgen pipeline.
        pythonEnv = pythonSet.mkVirtualEnv "ptwm-env" workspace.deps.all;

        # ── Rust toolchains ────────────────────────────────────────────
        rustStableTools = with pkgs; [
          rustc
          cargo
          rust-analyzer
          rustfmt
          clippy
        ];

        # Nightly Rust for rustdoc JSON (gated behind `-Z unstable-options`).
        rustNightly = fenixPkgs.minimal.toolchain;

        # Tooling shared across shells (excludes the Rust toolchain).
        coreTools = with pkgs; [
          uv
          git
          ninja
          pkg-config
          which
          prek
          maturin
        ];

        baseEnv = {
          UV_LINK_MODE = "copy";
          # Force uv to never download Python — it must use the Nix-built
          # interpreter we hand it via `UV_PYTHON`.
          UV_PYTHON_DOWNLOADS = "never";
          UV_PYTHON = "${pythonEnv}/bin/python";

          # Make pre-built wheels' dlopen() targets resolvable.
          LD_LIBRARY_PATH = libPath;
          NIX_LD_LIBRARY_PATH = libPath;
          NIX_LD = lib.fileContents "${pkgs.stdenv.cc}/nix-support/dynamic-linker";
        };

        # ── App helper ─────────────────────────────────────────────────
        appRuntimeInputs =
          [
            pythonEnv
            pkgs.maturin
          ]
          ++ rustStableTools;
        mkApp = name: extraInputs: text: {
          type = "app";
          program = "${
            pkgs.writeShellApplication {
              inherit name text;
              runtimeInputs = appRuntimeInputs ++ extraInputs;
            }
          }/bin/${name}";
        };

        # ── Docs site via pnpm.fetchDeps ───────────────────────────────
        pnpmDeps = pkgs.pnpm_9.fetchDeps {
          pname = "ptwm-docs";
          version = "0.1.0";
          src = ./docs/site;
          fetcherVersion = 2;
          # First build will emit the expected hash — fill it in below.
          hash = "sha256-t5rjigvy4VKJfGrXb8INUNNeYoshPLINPcR7UBMVAhA=";
        };
      in {
        devShells.default = pkgs.mkShell {
          name = "ptwm";

          packages = coreTools ++ rustStableTools ++ [pythonEnv];

          env = baseEnv;
        };

        devShells.paper = let
          # scheme-full guarantees all jmlr.cls transitive dependencies are
          # present without hunting them down one by one. It is already
          # cached in the Nix store on machines that run `nix develop .`.
          texEnv = pkgs.texlive.combined.scheme-full;
        in
          pkgs.mkShell {
            name = "ptwm-paper";

            packages = [
              texEnv
              pkgs.poppler_utils # pdfinfo, pdftotext
              pkgs.ghostscript # PDF manipulation utilities
            ];

            shellHook = ''
              alias build-paper='latexmk -pdf -bibtex -cd paper/lossless/latex/draft.tex'
              alias clean-paper='latexmk -C -cd paper/lossless/latex/draft.tex'
              echo "Paper shell ready. Commands:"
              echo "  build-paper   — compile paper/lossless/latex/draft.tex to PDF"
              echo "  clean-paper   — remove all generated files"
            '';
          };

        devShells.docs = pkgs.mkShell {
          name = "ptwm-docs";

          packages =
            coreTools
            ++ [
              rustNightly
              pythonEnv
            ]
            ++ (with pkgs; [
              nodejs_22
              pnpm_9
            ]);

          env = baseEnv;
        };

        # ── Workflow entry points ──────────────────────────────────────
        apps = {
          default = {
            type = "app";
            program = "${
              pkgs.writeShellApplication {
                name = "ptwm-check";
                runtimeInputs = appRuntimeInputs;
                text = ''
                  cargo test --workspace
                  maturin develop
                  ruff check python/ tests/
                  pyright
                  pytest tests/
                '';
              }
            }/bin/ptwm-check";
          };

          build = mkApp "ptwm-build" [] ''maturin develop "$@"'';
          build-release = mkApp "ptwm-build-release" [] ''maturin develop --release "$@"'';
          wheel = mkApp "ptwm-wheel" [] ''maturin build --release "$@"'';

          test = mkApp "ptwm-test" [] ''
            cargo test --workspace
            maturin develop
            pytest tests/ "$@"
          '';
          test-python = mkApp "ptwm-test-python" [] ''
            maturin develop
            pytest tests/ "$@"
          '';
          test-rust = mkApp "ptwm-test-rust" [] ''cargo test --workspace --all-targets "$@"'';

          lint = mkApp "ptwm-lint" [] ''ruff check python/ tests/ "$@"'';
          typecheck = mkApp "ptwm-typecheck" [] ''pyright "$@"'';
          check = mkApp "ptwm-check-full" [] ''
            ruff check python/ tests/
            pyright
            cargo test --workspace
            maturin develop
            pytest tests/
          '';

          clean = mkApp "ptwm-clean" [] ''
            rm -rf build/ dist/ target/ ./*.egg-info \
              python/ptwm/_core*.so .pytest_cache .ruff_cache
            cargo clean
          '';

          docgen =
            mkApp "ptwm-docgen"
            (
              [
                rustNightly
                pythonEnv
              ]
              ++ (with pkgs; [
                nodejs_22
                pnpm_9
              ])
            )
            ''
              cd docs/site
              pnpm install --frozen-lockfile
              pnpm run docgen
            '';

          docs-serve =
            mkApp "ptwm-docs-serve"
            (
              [
                rustNightly
                pythonEnv
              ]
              ++ (with pkgs; [
                nodejs_22
                pnpm_9
              ])
            )
            ''
              cd docs/site
              pnpm install --frozen-lockfile
              pnpm run docgen
              pnpm run dev "$@"
            '';
        };

        # ── Buildable artifacts ────────────────────────────────────────
        packages = {
          default = pkgs.rustPlatform.buildRustPackage {
            pname = "ptwm-core";
            version = "1.0.0";
            src = ./.;
            cargoLock.lockFile = ./Cargo.lock;

            cargoBuildFlags = [
              "-p"
              "ptwm-core"
            ];
            cargoTestFlags = [
              "-p"
              "ptwm-core"
            ];

            # `git2` with `vendored-openssl` builds openssl-src from
            # source, which calls `perl ./Configure`. Make perl available
            # in the sandbox so the build doesn't fail before cargo runs.
            nativeBuildInputs = [pkgs.perl];
          };

          # Static documentation site bundle. Output is under
          # $out/share/doc/ptwm/ (so it can be deployed by copying that
          # directory to any static host).
          docs = pkgs.stdenv.mkDerivation (_finalAttrs: {
            pname = "ptwm-docs";
            version = "0.1.0";
            src = ./.;

            nativeBuildInputs = [
              pkgs.nodejs_22
              pkgs.pnpm_9.configHook
              pkgs.rustPlatform.cargoSetupHook
              pythonEnv
              rustNightly
              pkgs.cacert
              # rustdoc-JSON extraction compiles ptwm-core, which pulls
              # openssl-src through git2/vendored-openssl. openssl-src's
              # build script calls `perl`, so it must be on PATH.
              pkgs.perl
            ];

            inherit pnpmDeps;
            pnpmRoot = "docs/site";

            # Vendor Cargo deps for the rustdoc-JSON extraction step
            # (`scripts/docgen/extract-rust.sh` invokes
            # `cargo rustdoc --package ptwm-core`).
            cargoDeps = pkgs.rustPlatform.importCargoLock {
              lockFile = ./Cargo.lock;
            };

            # Disable Nuxt telemetry and signal CI so no tool tries to
            # write to a non-existent TTY during the sandboxed build.
            NUXT_TELEMETRY_DISABLED = "1";
            CI = "1";
            # Treat docgen extraction failures as build failures rather
            # than soft warnings — we depend on the API pages.
            PTWM_DOCGEN_STRICT = "1";

            buildPhase = ''
              runHook preBuild
              export HOME=$TMPDIR
              cd docs/site
              pnpm run docgen
              pnpm run generate
              runHook postBuild
            '';

            installPhase = ''
              runHook preInstall
              mkdir -p $out/share/doc/ptwm
              cp -r .output/public/* $out/share/doc/ptwm/
              runHook postInstall
            '';
          });
        };

        treefmt = {
          programs = {
            alejandra.enable = true;
            deadnix.enable = true;
            shellcheck.enable = true;
            shfmt.enable = true;
            clang-format.enable = true;
            clang-tidy.enable = true;
            ruff.check = true;
            ruff.format = true;
            rustfmt.enable = true;
          };
          settings = {
            # Author-tooling templates contain Jinja-style placeholders
            # (`{{ kind }}`, `{{ author }}`, …) and are not valid source
            # in their own right. Skip every formatter on them.
            global.excludes = [
              "python/ptwm/ext_tooling/_templates/**"
              "docs/site/content/**"
            ];
            formatter = {
              shellcheck.options = [
                "-s"
                "bash"
              ];
              ruff-check.priority = 1;
              ruff-check.options = ["--fix-only"];
              ruff-format.priority = 2;
            };
          };
        };
      };
      flake = {};
    };
}
