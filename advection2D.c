/*******************************************************************************
2D advection example program which advects a Gaussian u(x,y) at a fixed vertical 
velocity and a variable horizontal velocity.



Outputs: initial.dat - inital values of u(x,y) 
         final.dat   - final values of u(x,y)
         vert_avg.dat - vertically averaged values of u(x,y)

         The output files initial and final have three columns: x, y, u
         The output file vert_avg have two columns: x, average y

         Compile with: gcc -fopenmp -o advection2D -std=c99 advection2D.c -lm

Notes: No directives to not use openmp if macro isn't present as program runs correctly
       in serial. 
       The time step is calculated using the CFL condition

********************************************************************************/

/*********************************************************************
                     Include header files 
**********************************************************************/

#include <stdio.h>
#include <math.h>

/*********************************************************************
                      Main function
**********************************************************************/

int main(){

  /* Grid properties */
  const int NX=1000;    // Number of x points
  const int NY=1000;    // Number of y points
  const float xmin=0.0; // Minimum x value
  const float xmax=30.0; // Maximum x value
  const float ymin=0.0; // Minimum y value
  const float ymax=30.0; // Maximum y value
  
  /* Parameters for the Gaussian initial conditions */
  const float x0=3.0;                    // Centre(x)
  const float y0=15.0;                    // Centre(y)
  const float sigmax=1.0;               // Width(x)
  const float sigmay=5.0;               // Width(y)
  const float sigmax2 = sigmax * sigmax; // Width(x) squared
  const float sigmay2 = sigmay * sigmay; // Width(y) squared

  /* Boundary conditions */
  const float bval_left=0.0;    // Left boudnary value
  const float bval_right=0.0;   // Right boundary value
  const float bval_lower=0.0;   // Lower boundary
  const float bval_upper=0.0;   // Upper bounary
  
  /* Time stepping parameters */
  const float CFL=0.9;   // CFL number 
  const int nsteps=800;  // Number of time steps

  /* Variable height velocity constants */
  const float frict_vel=0.2; // Friction velocity (u*)
  const float karman=0.41;   // Von Karman's constant (k)
  const float rough_len=1.0; // Roughness length  (z0)
  
  /* Velocity */
  const float vely=0.0; // Velocity in y direction
  float velx;           // Max velocity in x direction
  float velx_arr[NY+2]; // Variable velocities in the x direction
  
  /* Arrays to store variables. These have NX+2 elements
     to allow boundary values to be stored at both ends */
  float x[NX+2];            // x-axis values
  float y[NX+2];            // y-axis values
  float u[NX+2][NY+2];      // Array of u values
  float dudt[NX+2][NY+2];   // Rate of change of u
  float vert_vel_avg[NX+2]; // Average vertical velocity

  float x2;   // x squared (used to calculate iniital conditions)
  float y2;   // y squared (used to calculate iniital conditions)
  float ysum; // summation of y values (used to calcualte average vertical velocity)
  
  /* Calculate distance between points */
  float dx = (xmax-xmin) / ( (float) NX); // Distance between x points on the grid
  float dy = (ymax-ymin) / ( (float) NY); // Distance between y points on grid
  
  /* Calculate maximum velx */
  float maxy;                                          // The maximum height of y placed in the middle of the cell
  maxy = ((float) (NY+2) - 0.5) * dy;                  // Calculation for y value
  velx = (frict_vel / karman) * log(maxy / rough_len); // Calculation for maximum x velocity 
  
  /* Calculate time step using the CFL condition */
  /* The fabs function gives the absolute value in case the velocity is -ve */
  float dt = CFL / ( (fabs(velx / dx)) + (fabs(vely) / dy));
  
  /*** Report information about the calculation. ***/
  printf("Grid spacing dx     = %g\n", dx);
  printf("Grid spacing dy     = %g\n", dy);
  printf("CFL number          = %g\n", CFL);
  printf("Time step           = %g\n", dt);
  printf("No. of time steps   = %d\n", nsteps);
  printf("End time            = %g\n", dt*(float) nsteps);
  printf("Distance advected x = %g\n", velx*dt*(float) nsteps);
  printf("Distance advected y = %g\n", vely*dt*(float) nsteps);

  /*** Place x points in the middle of the cell.
       Can be parallelised as it affects independent indexes of 'x' array
       with independent calculations. ***/
  /* LOOP 1 */
#pragma omp parallel for
  for (int i=0; i<NX+2; i++){
    x[i] = ( (float) i - 0.5) * dx;
  }
  
  /*** Place y points in the middle of the cell.
       Can be parallelised as it affects independent indexes of 'y' array
       with independent calculations. ***/
  /* LOOP 2 */
#pragma omp parallel for
  for (int j=0; j<NY+2; j++){
    y[j] = ( (float) j - 0.5) * dy;
  }
 
  /*** Loop to calculate every value of velx according to height y.
       Can be parallelised as it does not share indexes and the velocity
       calculations are independent. ***/
#pragma omp parallel for
  for (int i=0; i<NY+2; i++){
  
    if (y[i] > rough_len){ // Velocity calculation only applies if y > roughness length
      velx_arr[i] = (frict_vel / karman) * log(y[i] / rough_len); // Calculation for velocity
    }
    else{ // Velocity set to 0 if y <= roughness length
      velx_arr[i] = 0.0;
    }
  }

  /*** Set up Gaussian initial conditions.
       Can be parallelised as each iteration of the loops alters a separate
       index of u with no dependencies between iterations. ***/
  /* LOOP 3 */
#pragma omp parallel for collapse(2) private(x2,y2)
  for (int i=0; i<NX+2; i++){
    for (int j=0; j<NY+2; j++){
      x2      = (x[i]-x0) * (x[i]-x0);
      y2      = (y[j]-y0) * (y[j]-y0);
      u[i][j] = exp( -1.0 * ( (x2/(2.0*sigmax2)) + (y2/(2.0*sigmay2)) ) );
    }
  }

  /*** Write array for initial u values out to file. ***/
  FILE *initialfile;
  initialfile = fopen("initial.dat", "w");
  
  /* LOOP 4 */
  /*** Data for the file needs to be written in order as no index is given to 
       later sort it, so it is best to leave it sequential. ***/
  for (int i=0; i<NX+2; i++){
    for (int j=0; j<NY+2; j++){
      fprintf(initialfile, "%g %g %g\n", x[i], y[j], u[i][j]);
    }
  }
  fclose(initialfile);
  
  /*** Update solution by looping over time steps. 
       Iterates over each time step and as each step requires the previous solution
       it can not be parallelised. ***/
  /* LOOP 5 */
  for (int m=0; m<nsteps; m++){
    
    /*** Apply boundary conditions at u[0][:] and u[NX+1][:] .
         Can be parallelised as each iteration affects a different index of u
         and boundary conditions are set independently. ***/
    /* LOOP 6 */
#pragma omp parallel for
    for (int j=0; j<NY+2; j++){
      u[0][j]    = bval_left;
      u[NX+1][j] = bval_right;
    }

    /*** Apply boundary conditions at u[:][0] and u[:][NY+1] .
         Can be parallelised as each iteration affects a different index of u
         and boundary conditions are set independently. ***/
    /* LOOP 7 */
#pragma omp parallel for
    for (int i=0; i<NX+2; i++){
      u[i][0]    = bval_lower;
      u[i][NY+1] = bval_upper;
    }
    
    /*** Calculate rate of change of u using leftward difference. 
         Can be parallelised as each iteration affects a different index of dudt
         and dudt is calculated independently from other indexes values. ***/
    /* Loop over points in the domain but not boundary values */
    /* LOOP 8 */
#pragma omp parallel for
    for (int i=1; i<NX+1; i++){
      for (int j=1; j<NY+1; j++){
        dudt[i][j] = -velx_arr[j] * (u[i][j] - u[i-1][j]) / dx
	            - vely * (u[i][j] - u[i][j-1]) / dy;
      }
    }
    
    /*** Update u from t to t+dt .
         Can be parallelised as each iteration affects a different index of u
         and dudt. The calculations are also independent of each other. ***/
    /* Loop over points in the domain but not boundary values */
    /* LOOP 9 */
#pragma omp parallel for
    for	(int i=1; i<NX+1; i++){
      for (int j=1; j<NY+1; j++){
	u[i][j] = u[i][j] + dudt[i][j] * dt;
      }
    }
    
  } // time loop
  
  /*** Write array of final u values out to file. 
       Loop writes the values in order so it has to remain sequential. ***/
  FILE *finalfile;
  finalfile = fopen("final.dat", "w");
  /* LOOP 10 */
  for (int i=0; i<NX+2; i++){
    for (int j=0; j<NY+2; j++){
      fprintf(finalfile, "%g %g %g\n", x[i], y[j], u[i][j]);
    }
  }
  fclose(finalfile);
  
  /*** Calcualte vertical averages and store in an array.
       Can be parallelised as indexes and calculations are independent 
       as long as ysum is private to each iteration. ***/
  /* Initialising file */
  FILE *vert_avg_file;
  vert_avg_file = fopen("vert_avg.dat", "w");
  
#pragma omp parallel for private(ysum)
  for (int i=0; i<NX+2; i++){
    ysum = 0;
    for (int j=1; j<NY+1; j++){
      ysum += u[i][j];
    }
    vert_vel_avg[i] = ysum/NY;
  }
  
  /*** Write array of average vertical velocities to file.
       Loop writes in order so it remains sequential. ***/
  for (int i=0; i<NX+2; i++){
    fprintf(vert_avg_file, "%g %g \n", x[i], vert_vel_avg[i]);
  }
  
  fclose(vert_avg_file);
  
  return 0;
}

/* End of file ******************************************************/

