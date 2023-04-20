Initial documentation on how to create a PDP2 model as a python script. **Please note that this file is incomplete.**

Creating a model
================

mdl\_name = MLPNetwork (net\_type = ...,
                       intervals = ...,
                       ticks\_per\_interval = ...
                      )

net_type: the type of network
 -expects an object of type MLPNetworkTypes: FEED\_FWD, SIMPLE\_REC, RBPTT, CONTINUOUS
intervals: the number of time intervals for which each example will be run
 -expects an integer
 -default value is 1
ticks_per_interval: the number of ticks, or subdivisions per time interval for CONTINUOUS networks only
 -expects an integer
 -default value is 1
 -for non-continuous networks, this should be set to 1

Adding layers to the network
============================

gp\_name = mdl\_name.group (units = ...,
                          group_type = [LIST],
                          input_funcs = [LIST],
                          output_funcs = [LIST],
                          label = ...
                          )

units: the number of units in the layer
 -expects an integer
group\_type: specifies whether the layer is input, output, hidden or bias
 -expects a list of values of type MLPGroupType: BIAS, INPUT, OUTPUT, HIDDEN
 -default value is HIDDEN
input\_funcs: specifies the functions used to compute the input to each unit of the group
 -expects a list of values of type MLPInputProcs: IN\_INTEGR, IN\_SOFT\_CLAMP, IN\_NONE
 -IN\_INTEGR integrates inputs over time so that they change gradually
 -IN\_SOFT\_CLAMP adds a factor to the input which pulls output values those of the input
 -default value is IN_NONE
output_funcs: specifies the functions to be applied to the output of each unit
 -expects a list of values of type MLPOutputProcs: OUT\_LOGISTIC, OUT\_INTEGR, OUT\_HARD\_CLAMP, OUT\_WEAK\_CLAMP, OUT\_BIAS, OUT\_NONE
 -OUT\_LOG\ISTIC computes the outputs of the unit 
 -OUT\_INTEGR integrates outputs over time so they change gradually
 -OUT\_HARD\_CLAMP clamps the output value to the input value
 -OUT\_WEAK\_CLAMP shifts the output value towards the input value by a certain amunt
 -OUT\_BIAS clamps the output of the unit to 1
 -bias units have OUT\_BIAS by default
 -input groups have OUT\_HARD\_CLAMP by default
 -hidden and output groups have OUT\_LOGISTIC by default
 -output groups also have OUT\_INTEGR by default unless there is an input integrator
label: takes a string label that can be used to refer to the group

Creating links betweent the layers
==================================

mdl\_name.link (pre\_link\_group = ...,
               post\_link\_group = ...,
               label = ...
              )
pre\_link\_group: the group from which the link originates
post\_link\_group: the group to which the link projects
label: takes a string label that can be used to refer to this group
 -links from bias unit to hidden and output groups are created by default

Create, read in and set parameters for example set
==================================================
//TO DO

Set other network parameters
============================

mdl\_name.set (num_presentations = ...,
              train_group_crit = ...,
              test_group_crit = ...,
              train_group_crit = ...,
              learning_rate = ...,
              weight_decay = ...,
              momentum = ...)

num\_presentations: the number of times you would like the example set to be presented in total
 -initially this sets num\_updates to the value specified in num\_presentations
 -default value is MLPConstants.DEF\_NUM\_UPDATES, which is 1
train\_group\_crit: the criterion used to stop training - an example is considered correct when the difference between output and target for each unit in the group is less than this value
test\_group\_crit: the criterion used to stop training - an example is considered correct when the difference between output and target for each unit in the group is less than this value
learning\_rate: weight changes are scaled by this amount
 -default value is MLPConstants.DEF_LEARNING_RATE, which is 0.1
weight\_decay: weights are reduced by this proportion after each weight update
 -default value is MLPConstants.DEF_WEIGHT_DECAY, which is 0
momentum: previous weight changes are carried over by this amount into the next step
 -default value is MLPConstants.DEF_MOMENTUM, which is 0.9

Set recording option
====================
//TO DO

Read the weights file
=====================
//TO DO

Train the network
=================

mdl\_name.train (update\_function = ...,
                num\_examples = ...,
                num\_updates = ...,
                reset\_examples = ...
               )

update\_function: the function to be used when weight updates are calculated
 -expects a value of type MLPUpdateFuncs: UPD_STEEPEST, UPD_MOMENTUM, UPD_DOUGSMOMENTUM
 -defaults to UPD_DOUGMOMENTUM
num\_examples: the number of examples to be run before a weight update
num\_updates: the total number of weight updates to be performed
 -if num\_examples is set, but num\_updates is not, num\_updates is re-calculated so that the entire example set is presented the number of times specified by num\_presentations
reset\_examples: if the number of examples run during the training phase (i.e. num\_examples * num\_updates) is smaller than the total number of examples in the example set, this parameter determines whether or not the next training phase will start from the next example in the set, or return to the first example
 -default value is False

Test the network
================

mdl\_name.test (num\_examples = ...,
               reset\_examples = ...
              )

num\_examples: the number of examples to be tested
reset\_examples: if num\_examples is smaller than the total number of examples in the example set, this parameter determines whether or not the next testing phase will start from the next example in the set, or return to the first example
 -default value is True

