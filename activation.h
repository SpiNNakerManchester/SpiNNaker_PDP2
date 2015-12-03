#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__

short_activ_t sigmoid       (net_t input);
short_activ_t sigmoid_prime (net_t input);
net_t         inv_sigmoid   (short_activ_t input);

#endif
