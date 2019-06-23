%% Generates Dataset for Shepp Logan Phantom Super Resolution 
close all
clear all
clc

N = 8*32;           % Set image resolution size (factor of 8)
N_dataset = 1;      % Number of randomly generated images
c_rate = 1/6;       % Compression rate in (0-1]


% [p,ellipse_base]=phantom(N);
% 
% ellipse_delta = zeros(10,6);
% 
% ellipse_delta(:,1) = .2*ellipse_base(:,1);
% ellipse_delta(1:2,2:3) = .01;
% ellipse_delta(3:5,2:3) = .05;
% ellipse_delta(6:10,2:3) = .005;
% ellipse_delta(3:10,4:5) = .05;
% ellipse_delta(3:10,6) = 20;
% 
% % display
% figure;
% imagesc(p);colormap bone; axis off;
% drawnow;

%% sequence of figure

% pack into video
sampleVideo_30 = VideoWriter('compressedSample_30.avi');
%sampleVideo_30.FrameRate = 10; % default 30
sampleVideo_30.Quality = 30; %default 75
open(sampleVideo_30);

sampleVideo_50 = VideoWriter('compressedSample_50.avi');
%sampleVideo_50.FrameRate = 10; % default 30
sampleVideo_50.Quality = 50; %default 75
open(sampleVideo_50);

sampleVideo_75 = VideoWriter('compressedSample_75.avi');
%sampleVideo_75.FrameRate = 10; % default 30
%sampleVideo_75.Quality = 75; %default 75
open(sampleVideo_75);

uncompressedVideo = VideoWriter('uncompressedSample.avi', 'Uncompressed AVI');
open(uncompressedVideo);


for i = 0:1999
    [p,ellipse_base]=phantom(seqGenerator(i),N);
    [rowM,colN]=size(p);
    for m = 1:rowM %make sure the value in p is between (0,1) to put in avi
        for n = 1:colN
            if p(m,n) < 0
                p(m,n) = 0;
            end
        end
    end
    
    ellipse_delta = zeros(10,6);

    ellipse_delta(:,1) = .2*ellipse_base(:,1);
    ellipse_delta(1:2,2:3) = .01;
    ellipse_delta(3:5,2:3) = .05;
    ellipse_delta(6:10,2:3) = .005;
    ellipse_delta(3:10,4:5) = .05;
    ellipse_delta(3:10,6) = 20;

    % % display
    % figure;
    % imagesc(p);colormap bone; axis off;
    % drawnow;

    writeVideo(sampleVideo_30,p);
    writeVideo(sampleVideo_50,p);
    writeVideo(sampleVideo_75,p);
    writeVideo(uncompressedVideo,p);
end

close(sampleVideo_30);
close(sampleVideo_50);
close(sampleVideo_75);
close(uncompressedVideo);