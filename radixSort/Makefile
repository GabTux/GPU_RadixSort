NVCC        = nvcc
DEB_FLAGS   = --generate-line-info
NVCC_FLAGS  = -O3 -std=c++17 --ptxas-options=--verbose -arch=compute_75 -code=sm_75 -I$(INC_DIR)
SRC_DIR     = src
INC_DIR     = include
EXE         = testRadix

all: compile

compile: $(EXE)

debug: NVCC_FLAGS += $(DEBFLAGS)
debug: compile

$(EXE):
	$(NVCC) -o $@ $(SRC_DIR)/main.cu $^ $(NVCC_FLAGS)

clean:
	rm -rf $(EXE)

run: $(EXE)
	./$(EXE)
