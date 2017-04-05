#!/usr/bin/env python

import numpy as np
from input_velodyne import *

"""Using raw velodyne 3D data, distinguish vehicle from raw lidar.
3) Training strategies: Compared to positive points on
vehicles, negative (background) points account for the majority
portion of the point cloud. Thus if simply pass all objectness
losses in (5) in the backward procedure, the network prediction
will significantly bias towards negative samples. To avoid
this effect, losses of positive and negative points need to be
balanced. Similar balance strategies can be found in Huang
et al. [11] by randomly discarding redundant negative losses.
In our training procedure, the balance is done by keeping all
negative losses but re-weighting them using
w1(p) = (
k|V|/(|P| − |V|) p ∈ P − V
1 p ∈ V
(7)
which denotes that the re-weighted negative losses are averagely
equivalent to losses of k|V| negative samples. In our case
we choose k = 4. Compared to randomly discarding samples,
the proposed balance strategy keeps more information of
negative samples.
Additionally, near vehicles usually account for larger portion
of points than far vehicles and occluded vehicles. Thus
vehicle samples at different distances also need to be balanced.
This helps avoid the prediction to bias towards near vehicles
and neglect far vehicles or occluded vehicles. Denote n(p) as
the number of points belonging to the same vehicle with p.
Since the 3D range scan points are almost uniquely projected
onto the point map. n(p) is also the area of the vehicle of p
on the point map. Denote n¯ as the average number of points
of vehicles in the whole dataset. We re-weight Lobj(p) and
Lbox(p) by w2 as
w2(p) = (
n/n ¯ (p) p ∈ V
1 p ∈ P − V
(8)
Using the losses and weights designed above, we accumulate
losses over deconv6a and deconv6b for the final training
loss
L =
X
p∈P
w1(p)w2(p)Lobj(p) + wbox X
p∈V
w2(p)Lbox(p) (9)
with wbox used to balance the objectness loss and the bounding
box loss."""
if __name__ == "__main__":
    pass
