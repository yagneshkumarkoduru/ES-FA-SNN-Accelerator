CC ?= gcc
CFLAGS ?= -O3 -Wall -Wextra -std=c99
LDFLAGS ?= -lm

TARGET = snn_simulator.exe
SRCS = main.c snn_engine.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.c snn_engine.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	del *.o *.exe *.json 2>nul || rm -f *.o *.exe *.json

run: $(TARGET)
	./$(TARGET) --cores 4 --neurons 128 --steps 10000 --sparsity 0.85
