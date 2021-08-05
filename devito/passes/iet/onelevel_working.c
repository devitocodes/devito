int Kernel(struct dataobj *restrict u_vec, const int time0_blk0_size, const int time_M, const int time_m, const int x0_blk0_size, const int x_M, const int x_>
{
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]][u_vec->size>

  for (int time0_blk0 = time_m; time0_blk0 <= time_M; time0_blk0 += time0_blk0_size)
  {
    /* Begin section0 */
    START_TIMER(section0)
    for (int x0_blk0 = x_m; x0_blk0 <= time_M - time_m + x_M; x0_blk0 += x0_blk0_size)
    {
      for (int y0_blk0 = y_m; y0_blk0 <= time_M - time_m + y_M; y0_blk0 += y0_blk0_size)
      {
        for (int time = time0_blk0, t0 = (time)%(2), t1 = (time + 1)%(2); time <= MIN(time_M, time0_blk0 + time0_blk0_size - 1); time += 1, t0 = (time)%(2), >
        {
          for (int x = MAX(x0_blk0, time + x_m); x <= MIN(x0_blk0 + x0_blk0_size - 1, time + x_M); x += 1)
          {
            for (int y = MAX(y0_blk0, time + y_m); y <= MIN(y0_blk0 + y0_blk0_size - 1, time + y_M); y += 1)
            {
              for (int z = z_m; z <= z_M; z += 1)
              {
                u[t1][-time + x + 1][-time + y + 1][z + 1] = u[t0][-time + x + 1][-time + y + 1][z + 1] + 1;
              }
            }
          }
        }
      }
    }
    STOP_TIMER(section0,timers)
    /* End section0 */
  }

  return 0;
}

