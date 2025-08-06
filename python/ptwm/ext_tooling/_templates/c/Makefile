NAME = {{ name }}
KIND = {{ kind }}

.PHONY: all clean
all: build/$(NAME).wasm

build/$(NAME).wasm: src/lib.c
	@mkdir -p build
	clang --target=wasm32-wasip1 -O3 -nostdlib \
	  -Wl,--no-entry -Wl,--export-all \
	  -o $@ $<

clean:
	rm -rf build
