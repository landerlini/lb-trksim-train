/* Floating point default type (32 or 64 bits) */
#define FLOAT_T float

/* Number of random nodes */
#ifndef N_RANDOM_NODES
#define N_RANDOM_NODES 128
#endif 

/* Number of input variables */
#ifndef N_X
#define N_X 5
#endif 

/* Number of output variables */
#ifndef N_OUT
#define N_OUT 9
#endif 

/* Lambda hyperparameter as defined in TrackPar/GanModel.py */
#ifndef LAMBDA
#define LAMBDA 0.3
#endif 

#define PIPELINE(__PIPENAME__, __NN__, __TX__, __TY__) \
extern "C" FLOAT_T* __NN__ (FLOAT_T *, const FLOAT_T*);                             \
extern "C" FLOAT_T* __TX__ (FLOAT_T *, const FLOAT_T*);                             \
extern "C" FLOAT_T* __TY__ (FLOAT_T *, const FLOAT_T*);                             \
                                                                                    \
extern "C"                                                                          \
FLOAT_T* __PIPENAME__ (FLOAT_T *ret, const FLOAT_T *input, const FLOAT_T *random)   \
{                                                                                   \
  short i;                                                                          \
  FLOAT_T bufferX [N_X + N_RANDOM_NODES];                                           \
  FLOAT_T bufferY [N_OUT];                                                          \
                                                                                    \
  __TX__ (bufferX, input);                                                          \
                                                                                    \
  for (i = 0; i < N_RANDOM_NODES; ++i)                                              \
    bufferX [N_X + i] = random[i];                                                  \
                                                                                    \
  __NN__ (bufferY, bufferX);                                                        \
                                                                                    \
  for (i = 0; i < N_OUT; ++i)                                                       \
    bufferY [i] = bufferY [i] * LAMBDA + random[N_RANDOM_NODES - N_OUT + i];        \
                                                                                    \
  __TY__ (ret, bufferY);                                                            \
                                                                                    \
  return ret;                                                                       \
}                                                                                   
                                                                                    


/* //////////// Define the pipelines for the various tracks /////////////// */

PIPELINE (reslong_pipe, reslong, reslong_tX, reslong_tY_inverse)   

PIPELINE (resupstream_pipe, resupstream, resupstream_tX, resupstream_tY_inverse)   

PIPELINE (resdownstream_pipe, resdownstream, resdownstream_tX, resdownstream_tY_inverse)   
    
    
    
    
    
    
    
    
    
    
    
    

