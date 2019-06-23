function output = Extractor(filename)

vidObj = VideoReader(filename);
vidHeight = vidObj.Height;
vidWidth = vidObj.Width;

output = zeros(vidHeight, vidWidth, vidObj.NumberOfFrames);

for img = 1:vidObj.NumberOfFrames;
    b = read(vidObj, img);
    output(:, :, img) = rgb2gray(b);
end