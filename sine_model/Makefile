CC = gcc
CFLAGS = -march=armv8.2-a+fp16 -lm

SRC = main.c
SRC_MICROKERNELS = $(wildcard ../microkernels/*.c)
OBJS = $(SRC:.c=.o) $(SRC_MICROKERNELS:.c=.o)

OUT = sine_model

INCLUDES = -I../microkernels

default: $(OUT)

$(OUT): $(OBJS)
	$(CC) -Wall -Wextra $(OBJS) -o $(OUT) $(CFLAGS) $(INCLUDES)

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf $(OUT) $(OBJS)