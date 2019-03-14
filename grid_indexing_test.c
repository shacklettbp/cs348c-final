#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const int GRID_SZ = 4;
int main() {
  char arr[GRID_SZ][GRID_SZ][GRID_SZ];
  memset(arr, 0, GRID_SZ*GRID_SZ*GRID_SZ);

  for (int z = 0; z < GRID_SZ; z++) {
    for (int y = 0; y < GRID_SZ; y++) {
      for (int x = 0; x < GRID_SZ / 2; x++) {
        int real_x = ((z + y) % 2) ? x * 2 : x * 2 + 1;
        arr[z][y][real_x] = (char)1;
      }
    }
  }

  for (int z = 0; z < GRID_SZ; z++) {
    for (int y = 0; y < GRID_SZ; y++) {
      for (int x = 0; x < GRID_SZ; x++) {
        printf("%c ", arr[z][y][x] == 1 ? 'X' : 'O');
      }
      printf("\n");
    }
    printf("\n");
  }

  printf("SWITCH\n\n");

  memset(arr, 0, GRID_SZ*GRID_SZ*GRID_SZ);

  for (int z = 0; z < GRID_SZ; z++) {
    for (int y = 0; y < GRID_SZ; y++) {
      for (int x = (z + y) % 2; x < GRID_SZ; x += 2) {
        arr[z][y][x] = (char)1;
      }
    }
  }

  for (int z = 0; z < GRID_SZ; z++) {
    for (int y = 0; y < GRID_SZ; y++) {
      for (int x = 0; x < GRID_SZ; x++) {
        printf("%c ", arr[z][y][x] == 1 ? 'X' : 'O');
      }
      printf("\n");
    }
    printf("\n");
  }


}
