int Kernel(struct dataobj *restrict u_vec, const int time0_blk0_size, const int time_M, const int time_m, const int x0_blk0_size, const int x0_blk1_size, const int x_M, const int x_m, const int y0_blk0_size, const int y0_blk1_size, const int y_M, const int y_m, const int z_M, const int z_m, const int nthreads, struct profiler * timers)
{
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]]) u_vec->data;

  for (int time0_blk0 = time_m; time0_blk0 <= time_M; time0_blk0 += 4)
  {
    /* Begin section0 */
    START_TIMER(section0)
    #pragma omp parallel num_threads(nthreads)
    {
      for (int x0_blk0 = x_m; x0_blk0 <= x_M + time_M; x0_blk0 += x0_blk0_size)
      {
        for (int y0_blk0 = y_m; y0_blk0 <= y_M + time_M; y0_blk0 += y0_blk0_size)
        {
          for (int time = time0_blk0, t0 = (time)%(2), t1 = (time + 1)%(2); time <= MIN(time_M, time0_blk0 + 4 - 1); time += 1, t0 = (time)%(2), t1 = (time + 1)%(2))
          {
	    #pragma omp for collapse(2) schedule(static,1)
            for (int x0_blk1 = MAX(x0_blk0,x_m + time); x0_blk1 <= MIN(x_M+time, x0_blk0 + x0_blk0_size - 1); x0_blk1 += x0_blk1_size)
            {
              for (int y0_blk1 = MAX(y0_blk0, y_m + time); y0_blk1 <= MIN(y_M+time, y0_blk0 + y0_blk0_size - 1); y0_blk1 += y0_blk1_size)
              {
                for (int x = x0_blk1; x <= MIN(x_M + time, x0_blk0 + x0_blk0_size - 1); x += 1)
                {
                  for (int y = y0_blk1; y <= MIN(y_M + time, y0_blk0 + y0_blk0_size - 1); y += 1)
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
        }
      }
    }
    STOP_TIMER(section0,timers)
    /* End section0 */
  }

  return 0;
}
