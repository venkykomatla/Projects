function I = DenoiseImage(A, net,G,TI,iter)
%denoiseImage Denoise an image using deep neural network
%
%   NOTE: This function requires the Deep Learning Toolbox.
%   
%   Class Support
%   -------------
%
%    In case of a single image, A can be a H-by-W or H-by-W-by-C matrix of
%    class uint8, uint16, single, double. In case of a batch of images, A
%    should be a H-by-W-by-C-by-N matrix of class uint8, uint16, single,
%    double where C denotes the number of image channels and N the batch
%    size. net is a network. B is a matrix of same class and shape as A.
%
%   Notes
%   -----
%
%   1. net can be obtained by using the function denoisingNetwork. It can
%      also be a user defined custom pre-trained denoising network.
%
%   2. It is expected that net would be trained to handle images with same
%      channel format (RGB or grayscale) as A.
%
%   3. The learning framework is assumed to be residual and the final
%      denoised image would be the difference of the input image and the estimated
%      residual output from the network layers.
%
%   4. The net is assumed to be a denoising network with a convolution layer
%      before the last regression layer and output of this function
%      corresponds to the response of this final convolution layer.
%

matlab.images.internal.errorIfgpuArray(A, net);
images.internal.requiresNeuralNetworkToolbox(mfilename);

narginchk(2,5);

validateInputImage(A);
validateInputNetwork(net);

channelCompatible = net.Layers(1).InputSize(3) == size(A,3) && net.Layers(end-1).NumFilters == size(A,3);
if ~channelCompatible
    error(message('images:denoiseImage:incompatibleImageNetwork'));
end

classOfA = class(A);
if ~isa(A,'single')
    inputImage = im2single(A);
else
    inputImage = A;
end

numLayers = size(net.Layers,1);
res = activations(net,inputImage,numLayers-1,'OutputAs','channels');
res(isnan(res))=0;
res=res+rescale(inputImage,min(G(:)),max(G(:)))-(((TI/iter)*5e-2*rand(size(G)))+G);
I = rescale(inputImage,min(G(:)),max(G(:))) - res;

% if isinteger(A)   
%     I = cast(double(intmax(classOfA))*I,classOfA);
% elseif isa(A,'double')
%     I = cast(I,'double');
% end

end

function validateInputImage(A)
supportedClasses = {'uint8','uint16','single','double'};
attributes = {'nonempty','nonsparse','real','nonnan','finite'};

validateattributes(A, supportedClasses, attributes, mfilename, 'A');
if ndims(A) > 4
    error(message('images:denoiseImage:invalidImageFormat'));
end
end

function validateInputNetwork(net)
supportedClasses = {'SeriesNetwork','DAGNetwork'};
attributes = {'nonempty','nonsparse'};
validateattributes(net, supportedClasses, attributes, mfilename, 'net');
validateattributes(net.Layers(end), {'nnet.cnn.layer.RegressionOutputLayer'}, attributes, mfilename, 'net');
if ~isa(net.Layers(end-1),'nnet.cnn.layer.Convolution2DLayer')
    error(message('images:denoiseImage:lastLayerNotConv2d'));
end
end


