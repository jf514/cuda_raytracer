################################################################################
# Compilation flags for this toy rendering project, chopped down from the 
# Makefile found in the CUDA samples
################################################################################

################################################################################
# General commands
CUDA_PATH ?= /usr/local/cuda
HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

################################################################################
# Internal flags

# Compile for 64 bit architecture
NVCCFLAGS   := -m64
CCFLAGS     :=
LDFLAGS     :=

# Debug flags - uncomment for debug, debug w/ cuda-gdb
#NVCCFLAGS += -g -G

# General flags
ALL_CCFLAGS := -I ../common -I ../dependencies
ALL_CCFLAGS += --threads 0 --std=c++17
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)

# Flags passed to host compiler
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

# General linker flags
ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)

# Flags passed to general linker
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  := -I../../../Common
LIBRARIES :=

################################################################################
# Gencode arguments - these indicate particular GPU targets for which to generate
# code.
ifeq ($(GENCODE_FLAGS),)
    # Specific architectures to target
    SMS ?= 50 52 60 61 70 75 80 86 89 90
    # Generate SASS code for each SM architecture listed in $(SMS)
    $(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
    # Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
    HIGHEST_SM := $(lastword $(sort $(SMS)))
    ifneq ($(HIGHEST_SM),)
        GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
    endif
endif

################################################################################
# Target rules
all: build

build: scene

rebuild: clean build

parallel.o: ../common/parallel.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

scene.o:scene.cu ../common/*.*
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

scene: scene.o parallel.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

run: build
	$(EXEC) ./scene

clean:
	rm -f scene scene.o
