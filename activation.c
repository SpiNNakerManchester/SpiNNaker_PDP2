// SpiNNaker API
#include "spin1_api.h"

// mlp
#include "mlp_params.h"
#include "mlp_types.h"
#include "mlp_macros.h"

#include "activation_lut.h"
#include "activation.h"

//The sigmoid routine computes the sigmoid function with interpolation.
//the input parameter is a 5.27 fixed point 32-bit value. The returned output
//is a 1.15 fixed point 16-bit value.
//the output values are stored in a table with 256 elements. Since the function
//is symmetric with respect to the value (x=0, y=0.5), the values stored in the
//table are only the ones related to x >= 0. The values for x < 0 are computed
//as 1 - f(-x)
//The interpolation is computed linearly using the lowest set of bits of the
//input value
activation_t sigmoid (net_t input)
{
  #ifdef SPINN_SIGMD_ROUNDI
    // round input
    net_t temp = input + (net_t) (1 << (SPINN_SIGMD_LUT_SHIFT - 1));
  #else // truncate input (default)
    net_t temp = input;
  #endif

  // check if outside the LUT range
  if (temp >= (net_t) SPINN_SIGMD_MAX_INPUT)
  {
    return ((activation_t) SPINN_ACTIV_MAX);
  }
  else if (temp <= (net_t) SPINN_SIGMD_MIN_INPUT)
  {
    return ((activation_t) SPINN_ACTIV_MIN);
  }
  else
  // the input value is inside the range of the lookup table. The value needs
  // to be interpolated appropriately
  {
    uchar x0;             // input bits to access LUT
    long_activ_t z;       // input remainder bits to do interpolation
    activation_t y0, y1;  // look-up values

    // LUT contains only positive inputs
    if (temp >= 0)
    {
      x0 = temp >> SPINN_SIGMD_LUT_SHIFT; //input value of the lookup table
      z  = (long_activ_t) temp & (long_activ_t) SPINN_SIGMD_LUT_IMASK;
    }
    else
    {
      x0 = (-temp) >> SPINN_SIGMD_LUT_SHIFT; //input value of the lookup table
      z  = (long_activ_t) (-temp) & (long_activ_t) SPINN_SIGMD_LUT_IMASK;
    }

    y0 = sigmoid_lut[x0]; // value corresponding to the lookup table

    // if x0 is largest value in table -- interpolate with MAX value
    if (x0 == (SPINN_SIGMD_RES - 1))
      y1 = (activation_t) SPINN_ACTIV_MAX;
    else
      y1 = sigmoid_lut[x0 + 1];

    // interpolate using long variables and round off
    // s0.15 = ((s0.15 - s0.15) * s0.22) >> 22
    long_activ_t out_tmp = (((long_activ_t) (y1 - y0) * z)
                             + (long_activ_t) (1 << (SPINN_SIGMD_LUT_SHIFT - 1)))
	                     >> SPINN_SIGMD_LUT_SHIFT;

    activation_t output = y0 + (activation_t) out_tmp;
          
    // return the symmetric result for negative inputs!
    if (temp < 0)
      return ((activation_t) (1 << SPINN_ACTIV_SHIFT) - output);
    else
      return (output);
  }
}

//The inv_sigmoid routine computes the inverse of the sigmoid function with interpolation.
//the input parameter is a 1.15 fixed point 16-bit value. The returned output
//is a 5.27 fixed point 32-bit value.
//the output values are stored in a table with 256 elements. Since the function
//is symmetric with respect to the value (x=0.5, y=0), the values stored in the
//table are only the ones related to 0.5 <= x <= 1. The values for 0 <= x < 0.5
//are computed as -f(ABS(x - 1))
//The interpolation is computed linearly using the lowest set of bits of the
//input value
net_t inv_sigmoid (activation_t input)
{
  // check if outside the LUT range
  if (input >= (activation_t) SPINN_ACTIV_MAX)
  {
    return ((net_t) SPINN_SIGMD_MAX_INPUT);
  }
  else if (input <= (activation_t) SPINN_ACTIV_MIN)
  {
    return ((net_t) SPINN_SIGMD_MIN_INPUT);
  }
  else // evaluate logistic inverse function
  {
    activation_t input_adapted; //input value for the lookup table
    long_net_t temp; //variable for interpolation
    uchar x0; //first input value for interpolation
    net_t y0, y1; //output values for interpolation

    // LUT covers input range [0.5, SPINN_ACTIV_MAX)
    // limit input to this range
    input_adapted = ABS(input - (1 << (SPINN_ACTIV_SHIFT - 1)));

    x0 = (input_adapted >> SPINN_INVSIG_LUT_SHIFT);

    y0 = inv_sigmoid_lut[x0];

    // if x0 is largest value in table -- interpolate with MAX value
    if (x0 == (SPINN_SIGMD_RES - 1))
      y1 = (net_t) SPINN_SIGMD_MAX_INPUT;
    else
      y1 = inv_sigmoid_lut[x0 + 1];
        
    // interpolate (using long variables)
    temp = (long_net_t) y0
             + (((long_net_t) (input_adapted & SPINN_INVSIG_LUT_IMASK)
             * (long_net_t) (y1 - y0)) >> SPINN_INVSIG_LUT_SHIFT);

    //NOTE: need to adjust fixed-point position due to change in representation.
    //TODO: re-work look-up table with new fixed-point position!
    temp = temp >> 4;

    // saturate output -- should never be applied
    // no need to check for negative values!
    if (temp > (net_t) SPINN_SIGMD_MAX_INPUT)
      temp = (net_t) SPINN_SIGMD_MAX_INPUT;

    // if input < 0.5 return symmetric value
    if (input < (1 << (SPINN_ACTIV_SHIFT - 1)))
      return (-temp);
    else
      return (temp);
  }
}
