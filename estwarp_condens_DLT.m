function param = estwarp_condens_DLT(frm, param, opt)

global useGpu;
n = opt.numsample;
%sz = size(tmpl.mean);
%N = sz(1)*sz(2);

if ~isfield(param,'param')
  param.param = repmat(affparam2geom(param.est(:)), [1,n]);
else
  cumconf = cumsum(param.conf);
  idx = floor(sum(repmat(rand(1,n),[n,1]) > repmat(gather(cumconf),[1,n])))+1;
  param.param = param.param(:,idx);
end
param.param = param.param + randn(6,n).*repmat(opt.affsig(:),[1,n]); %+ repmat([opt.motion, 0, 0, 0, 0]',[1,n]) ;
% bbox: n-by-5
bbox = param2bbox(param.param, size(frm(:,:,1)), [227, 227]);

input_data = zeros(227, 227, 3, n);
d = load('./caffe/ilsvrc_2012_mean');
IMAGE_MEAN = d.image_mean;

for i = 1:n
	rect = bbox(i, 1:4);
	im = frm(rect(2):rect(2)+rect(4), rect(1):rect(1)+rect(3), :);
	im = imresize(im, [227, 227], 'bilinear');
	input_data(:,:,:,i) = im(:,:,[3 2 1]) - IMAGE_MEAN;
end

tic;
% scores: 
scores = caffe('forward', input_data);
toc;
confidence = reshape(scores{1}, [21, n]);

%{
% create crop_size x crop_size x particle_num matrix, to be passed into NN
%wimgs = warpimg(frm, affparam2mat(param.param), sz);
if useGpu
    data = gpuArray(reshape(wimgs,[N,n]));
else
    data = reshape(wimgs,[N,n]);
end

t = nnff(nn, data', zeros(n, 1));
confidence = t.a{6}';
%}

disp(max(confidence));
if max(confidence) < opt.updateThres
    param.update = true;
else
    param.update = false;
end
confidence = confidence - min(confidence);
param.conf = exp(double(confidence) ./opt.condenssig)';
param.conf = param.conf ./ sum(param.conf);
[maxprob,maxidx] = max(param.conf);
if maxprob == 0 || isnan(maxprob)
    error('overflow!');
end
param.est = affparam2mat(param.param(:,maxidx));
param.wimg = reshape(data(:,maxidx), sz);

if exist('coef', 'var')
    param.bestCoef = coef(:,maxidx);
end
