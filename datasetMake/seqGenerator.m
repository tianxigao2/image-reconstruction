%% sequence generator:
% generate sequence of similar matrix E to pass in phantom.m file
% phantom generated will be slightly different from org phantom

% use toft --modified shepp logan phantom as default
function toft = seqGenerator(s)    % in makeFile loop, s will count from 0 to 1999

%         A    a     b    x0    y0    phi
%        ---------------------------------
toft = [  1   .69   .92    0     0     0   
        -.8  .6624 .8740   0  -.0184   0
        -.2  .1100 .3100  .22    0    -18
        -.2  .1600 .4100 -.22    0     18
         .1  .2100 .2500   0    .35    0
         .1  .0460 .0460   0    .1     0
         .1  .0460 .0460   0   -.1     0
         .1  .0460 .0230 -.08  -.605   0 
         .1  .0230 .0230   0   -.606   0
         .1  .0230 .0460  .06  -.605   0   ];

 % To make change slight, scale the parameter
 %         A        a           b           x0      y0      phi
 %       -----------------------------------------------------------
 prop = [0.00001    0.000001    0.0000001   0.00001 0.00001 pi/3600000];
 prop = s .* prop;
 for i = 2:10
     toft(i, :) = prop + toft(i, :);
 end
 
       
