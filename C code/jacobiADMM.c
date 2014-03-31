/*
 * Solve a distributed lasso problem, i.e.,
 *    
 *       min ||x||_1
 *       s.t. Ax = b
 * 
 * The implementation uses MPI for distributed communication
 * and the GNU Scientific Library (GSL) for BLAS operations. 
 *
 * Reference: W. Deng, M.-J. Lai, Z. Peng, and W. Yin, Parallel Multi-Block ADMM with o(1/k) Convergence, UCLA CAM 13-64, 2013. 
 * Date:     Jan 20, 2014
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "mmio.h"
#include <mpi.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

void soft_threshold(gsl_vector *v, double k); //soft thresholding function

// main function
int main(int argc, char **argv) {

  int rank;
  int size;

  MPI_Init(&argc, &argv);               // Initialize the MPI execution environment
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Determine current running process
  MPI_Comm_size(MPI_COMM_WORLD, &size); // Total number of processes


  /* stopping criterions */
  const int MAX_ITER  = 1000;   // maximum number of iterations
  const double tol    = 1e-6;   // relative error tolerance
  FILE *test;                   // write results to file
  int m;                        // # of rows 
  int n;                        // # of columns
  
  m = 1000;                   // you can set it to a smaller size for fun
  n = 2000/size;
  
  double entry, startTime, endTime, commStartTime, commEndTime, totalStartTime, totalEndTime;
  char s[100];

  int row, col;    
  gsl_rng * rg;
  const gsl_rng_type* T;
  gsl_rng_env_setup();
  T = gsl_rng_default;
  rg = gsl_rng_alloc(T);

  /* Generate matrix A */
  if(rank==0)
  {
    printf("======================\n");
    printf("Start to generate A \n");
    startTime = MPI_Wtime();
  }
  
  gsl_matrix *A = gsl_matrix_calloc(m, n);
  // entries of matrix A from Gaussian distribution
  for (int i = 0; i < m*n; i++) 
  {
    row = i % m;
    col = floor(i/m);
    entry = gsl_ran_gaussian(rg, 1.0);
    gsl_matrix_set(A, row, col, entry);
  }
  
  if(rank==0)
  {
    printf("A is generated!\n");
    endTime = MPI_Wtime();
    printf("it takes %lf s to generate A! \n", endTime - startTime);
  }

  /* generate a sparse vector xs */
  gsl_vector *xs = gsl_vector_calloc(n);
  int k = 0.05*n;    // k is the # of non-zeros
  int id;            // id saves the location of the non-zero component;
  
  if(rank==0)
  {
    printf("======================\n");
    printf("Start to generate xs \n");
  }
  // generate the sparse vector xs
  for(int j=0; j<k;j++)
  {
    entry = gsl_ran_gaussian(rg, 1.0);
    id = rand()%n;
    if(entry>0) 
      gsl_vector_set(xs,  id, entry);
    else
      gsl_vector_set(xs,  id, entry);
  }
  
  if(rank==0) printf("xs is generated! \n");
  
  
  //caculate b  
  gsl_vector *tmpb = gsl_vector_calloc(m);
  gsl_vector *b = gsl_vector_calloc(m);
  gsl_blas_dgemv(CblasNoTrans, 1, A, xs, 0, tmpb); // Axs = b
  MPI_Allreduce(tmpb->data, b->data,  m, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  gsl_vector_free(tmpb);
  if(rank==0) printf("b is generated! \n");
  MPI_Barrier(MPI_COMM_WORLD);

  /*
   * These are all variables related to ADMM itself. There are many
   * more variables than in the Matlab implementation because we also
   * require vectors and matrices to store various intermediate results.
   * The naming scheme follows the Matlab version of this solver.
   */ 
  
  double rho = 10.0/gsl_blas_dasum(b);
  double tau = 0.1 * size * rho;
  double gamma = 1.0;
  
  double dlam_nrm, lower_bnd, dx_nrm, cross_term;
  gsl_vector *x_old      = gsl_vector_calloc(n);  // previous x
  gsl_vector *x          = gsl_vector_calloc(n);  // current x
  gsl_vector *x_new      = gsl_vector_calloc(n);  // newly updated x
  gsl_vector *res_old    = gsl_vector_calloc(m);  // previous residual
  gsl_vector *res_new    = gsl_vector_calloc(m);  // newly updated residual
  gsl_vector *res        = gsl_vector_calloc(m);  // current residual 
  gsl_vector *foo        = gsl_vector_calloc(m);  // temporary variable
  gsl_vector *lambda     = gsl_vector_calloc(m);  // dual variable
  gsl_vector *tmp        = gsl_vector_calloc(n);  // temporary variable 
  gsl_vector *local_Ax   = gsl_vector_calloc(m);  // the A_i x_i computed in each machine
  gsl_vector *global_Ax  = gsl_vector_calloc(m);  // the sum A_i x_i
  gsl_vector *grad       = gsl_vector_calloc(n);  // gradient 
  gsl_vector *z          = gsl_vector_calloc(m);  // temporary variable 
  gsl_vector *x_diff     = gsl_vector_calloc(n);  // the difference between x and xs
  
  double send[3]; // an array used to aggregate 3 scalars at once
  double recv[3]; // used to receive the results of these aggregations
  
  gsl_blas_ddot(xs, xs, &send[1]); // compute the 2-norm of xs
  
  /* Main Jacobi ADMM solver loop */
  int iter = 0;
  if (rank == 0) 
  {
    printf("%3s %10s \n", "#", "relative error");
    sprintf(s, "results/test.m");      // read results into a test.m file
    test = fopen(s, "w");
    fprintf(test,"res = [ \n");
  }
  
  gsl_vector_memcpy(res_new, b);
  gsl_vector_scale(res_new, -1.0);
  
  totalStartTime = MPI_Wtime();      // record the starting point of total time
  while (iter < MAX_ITER) {

    startTime = MPI_Wtime();      // record the starting point of a single iteration
    // update x, x_old
    gsl_vector_memcpy(x_old, x);
    gsl_vector_memcpy(x, x_new);
    // update res, res_old
    gsl_vector_memcpy(res_old, res);
    gsl_vector_memcpy(res, res_new);
    
    /* x-update:  */
    // calculate gradient
    gsl_vector_memcpy(foo, res_new);
    gsl_vector_scale(foo, rho);	  
    gsl_vector_memcpy(z, foo);
    gsl_vector_sub(z, lambda);
    
    gsl_blas_dgemv(CblasTrans, 1, A, z, 0, grad);
    
    
    gsl_vector_memcpy(tmp, grad);
    gsl_vector_scale(tmp, -1.0/tau);
    gsl_vector_add(tmp, x_new);
    // apply the shrinkage operator
    soft_threshold(tmp, 1.0/tau);
    gsl_vector_memcpy(x_new, tmp);
    
    /* lambda-update: */
    // calculate residual
    gsl_blas_dgemv(CblasNoTrans, 1, A, x_new, 0, local_Ax);
    
    commStartTime = MPI_Wtime();
    MPI_Allreduce(local_Ax->data, global_Ax->data,  m, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);	  
    commEndTime = MPI_Wtime();
    
    gsl_vector_memcpy(res_new, global_Ax);
    gsl_vector_sub(res_new, b);
    
    gsl_vector_memcpy(foo, res_new);
    gsl_vector_scale(foo, gamma * rho);
    gsl_vector_sub(lambda, foo);
    
    //calculate the difference of two iterations
    gsl_vector_memcpy(x_diff, x);
    gsl_vector_sub(x_diff, x_new);
    
    // check stopping criterion
    
    gsl_blas_ddot(x_diff, x_diff, &send[0]);
    gsl_blas_ddot(x_new, x_new, &send[1]);
    send[2] = gsl_blas_dasum(x);
    
    MPI_Allreduce(send, recv, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    
    if(sqrt(recv[0])< tol * sqrt(recv[1]))
      break;

    /* Dynamically update tau: proximal parameter*/
    gsl_vector_memcpy(foo, res);
    gsl_vector_sub(foo, res_new);
    
    gsl_blas_ddot(res_new, foo, &cross_term);
    cross_term = 2 * rho * cross_term;
    dx_nrm = tau * recv[0];
    gsl_blas_ddot(res_new, res_new, &dlam_nrm);
    dlam_nrm = (2 - gamma) * rho * dlam_nrm;
    lower_bnd = dx_nrm + dlam_nrm + cross_term;
    if(lower_bnd < 0)
    {
      tau = tau * 2;
      gsl_vector_memcpy(foo, res_new);
      gsl_vector_scale(foo, rho* gamma);
      gsl_vector_add(lambda, foo);
      
      gsl_vector_memcpy(res_new, res);
      gsl_vector_memcpy(res, res_old);
      
      gsl_vector_memcpy(x_new, x);
      gsl_vector_memcpy(x, x_old);
      
    }
    else if(iter%10 == 0)
      tau = tau * 0.5;
    
    // compute the relative error of x
    gsl_vector_memcpy(x_diff, x_new);
    gsl_vector_sub(x_diff, xs);
    gsl_blas_ddot(x_diff, x_diff, &send[0]);
    gsl_blas_ddot(xs, xs, &send[1]);
    MPI_Allreduce(send, recv, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    if (rank == 0) 
    {
      endTime = MPI_Wtime();
      printf("%3d %e %10.4f\n", iter,
	     sqrt(recv[0])/sqrt(recv[1]), recv[2]);
      fprintf(test, "%e %e %e %e;\n", sqrt(recv[0])/sqrt(recv[1]), recv[2],
	      endTime - startTime, commEndTime - commStartTime);
    }
    iter++;
    
  }
  
  /* Have the master write out the results to disk */
  if(rank==0){
    totalEndTime =MPI_Wtime();
    fprintf(test, "]; \n");
    fprintf(test,"semilogy(1:length(res),res); \n");
    fprintf(test,"xlabel('# of iteration'); ylabel('||x - xs||/||xs||');\n");
    printf("Elapsed time is: %lf \n", totalEndTime - totalStartTime);
  }
 
  MPI_Finalize(); /* Shut down the MPI execution environment */
  
  /* Clear memory */
  gsl_matrix_free(A);
  gsl_vector_free(b);
  gsl_vector_free(x);
  gsl_vector_free(z);
  gsl_vector_free(x_new);
  gsl_vector_free(x_old);
  gsl_vector_free(foo);
  gsl_vector_free(tmp);
  gsl_vector_free(res_new);
  gsl_vector_free(res_old);
  gsl_vector_free(x_diff);
  gsl_vector_free(local_Ax);
  gsl_vector_free(global_Ax);
  gsl_vector_free(res);
  gsl_vector_free(grad);
  gsl_vector_free(lambda);

  return EXIT_SUCCESS;
}

void soft_threshold(gsl_vector *v, double k) 
{
  double vi;
  for (int i = 0; i < v->size; i++) 
  {
    vi = gsl_vector_get(v, i);
    if (vi > k)       { gsl_vector_set(v, i, vi - k); }
    else if (vi < -k) { gsl_vector_set(v, i, vi + k); }
    else              { gsl_vector_set(v, i, 0); }
  }
}
