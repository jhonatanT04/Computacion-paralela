# ============================================================================
# PRÁCTICA 1 - Convolución CPU vs GPU
# RTX 4060 Laptop GPU = Ada Lovelace = sm_89
# ============================================================================

OPENCV_INC = -I/home/justin/Documentos/opencv/opencvi/include/opencv4/
OPENCV_LIB = -L/home/justin/Documentos/opencv/opencvi/lib/
OPENCV_LINK = -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui \
              -lopencv_videoio -lopencv_objdetect -lopencv_ml \
              -lopencv_cudaobjdetect -lopencv_cudaimgproc -lopencv_cudawarping \
              -lopencv_cudaarithm -lpthread -lboost_system

# Compilar versión paralela (GPU)
gpu: filtro_paralelo.cu
	nvcc filtro_paralelo.cu -std=c++17 -arch=sm_89 \
	$(OPENCV_INC) $(OPENCV_LIB) $(OPENCV_LINK) \
	-o visionGPU.bin

# Compilar versión secuencial (CPU)
cpu: filtro_secuencial.cpp
	g++ filtro_secuencial.cpp -std=c++17 -O2 \
	$(OPENCV_INC) $(OPENCV_LIB) $(OPENCV_LINK) \
	-o visionCPU.bin

# Compilar ambas
all: cpu gpu
	@echo "Compilación completada: visionCPU.bin y visionGPU.bin"

# Ejecutar GPU
runGPU:
	LD_LIBRARY_PATH=/home/justin/Documentos/opencv/opencvi/lib/:$$LD_LIBRARY_PATH \
	./visionGPU.bin $(IMG) $(MASK) $(OUT)

# Ejecutar CPU
runCPU:
	LD_LIBRARY_PATH=/home/justin/Documentos/opencv/opencvi/lib/:$$LD_LIBRARY_PATH \
	./visionCPU.bin $(IMG) $(MASK) $(OUT)

clean:
	rm -f visionCPU.bin visionGPU.bin

.PHONY: all gpu cpu runGPU runCPU clean