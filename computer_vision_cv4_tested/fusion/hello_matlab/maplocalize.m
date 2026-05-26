%%%%%%%%%%%%%% Read Peter Corke Chapter 6 on EKF free in Merriam Library
% Follow along
%
% Look up MATLAB syntax you are not familiar with
%
% Install RCV2 based on https://petercorke.com/toolboxes/robotics-toolbox/
%
% edit RCV2-copy\rcvtools\robot\plot_vehicle.m
%
% Change line with “plot_poly” to h = plot_poly(corners, 'animate', 'axis', opt.axis, args{:});
%
% h = plot_poly(corners, 'animate', args{:});
%
% clear functions
% clear classes
% rehash toolboxcache


%%%%%%%%%%%%%%%%%%%%%%%%% Simulate vehicle path
V = diag([0.02, 0.5*pi/180].^2);
veh = Bicycle('covar', V)
odo = veh.step(1, 0.3)
veh.x'
veh.f([0 0 0], odo)
veh.add_driver(RandomPath(10))
veh.run()

%%%%%%%%%%%%%%%%%%%%%%%%%%% Generate Fig 6.4
veh.Fx( [0,0,0], [0.5,0.1])
P0=diag([0.005, 0.005, 0.001].^2);
ekf=EKF(veh, V, P0);
ekf.run(1000);
veh.plot_xy()
hold on
ekf.plot_xy('r')
P700 = ekf.history(700).P
sqrt(P700(1,1))
ekf.plot_ellipse('g')

%%%%%%%%%%%%%%%%%%%%%% Figure 6.5
%%% ekf.plot_P();


%%%%%%%%%%%%%%%%%%%%%%%% Map landmark demo
map=LandmarkMap(20, 10)
map.plot()
W=diag([0.1,1*pi/180].^2);
sensor=RangeBearingSensor(veh, map, 'covar', W)
[z, i]=sensor.reading()
%landmark(17), should be map(17)


%%%%%%%%%%%%%% Generates Fig 6.7a
map=LandmarkMap(20);
veh = Bicycle('covar', V);
veh.add_driver(RandomPath(map.dim));
sensor=RangeBearingSensor(veh, map, 'covar', W, 'angle', [-pi/2 pi/2], 'range', 4, 'animate');
ekf=EKF(veh, V, P0, sensor, W, map);
ekf.run(1000);
map.plot()
veh.plot_xy();
ekf.plot_xy('r');
ekf.plot_ellipse('k')


%%%%%%%%%%%%%%%% Generates Fig 6.9 example
map = LandmarkMap(20)
veh =  Bicycle();
veh.add_driver(RandomPath(map.dim));
W = diag([0.1, 1*pi/180].^2);
sensor = RangeBearingSensor(veh, map, 'covar', W);
ekf = EKF(veh, [], [], sensor, W, []);
ekf.run(1000)

map.plot();
ekf.plot_map('g');
veh.plot_xy('b');

ekf.landmarks(:,6)
ekf.x_est(19:20)'
ekf.P_est(19:20,19:20)


%%%%%%%%%%%%%%%%%% Generates Fig 6.11 example

P0 = diag([.01, .01, 0.005].^2);
map = LandmarkMap(20);
veh = Bicycle('covar', V);
veh.add_driver(RandomPath(map.dim));
sensor=RangeBearingSensor(veh, map, 'covar', W);
ekf=EKF(veh, V, P0, sensor, W, []);
ekf.run(1000);

map.plot();
ekf.plot_map('g');
ekf.plot_xy('r');
veh.plot_xy('b');
