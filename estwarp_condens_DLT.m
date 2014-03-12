function param = estwarp_condens_DLT(frame, param, opt)

global useGpu;
global caffe_batch_size;

n = opt.numsample;
sz = size(frame(:,:,1));

if ~isfield(param,'param')
  param.param = repmat(affparam2geom(param.est(:)), [1,n]);
else
  cumconf = cumsum(param.conf);
  idx = floor(sum(repmat(rand(1,n),[n,1]) > repmat(gather(cumconf),[1,n])))+1;
  param.param = param.param(:,idx);
end
param.param = param.param + randn(6,n).*repmat(opt.affsig(:),[1,n]); %+ repmat([opt.motion, 0, 0, 0, 0]',[1,n]) ;
% bbox: n-by-5
bbox = param2bbox(param.param, size(frame(:,:,1)), [227, 227]);

images = zeros(227, 227, 3, n, 'single');
d = load('./caffe/ilsvrc_2012_mean');
IMAGE_MEAN = d.image_mean;
IMAGE_MEAN = imresize(IMAGE_MEAN, [227, 227], 'bilinear');

% images: 227 x 227 x 3 x n
% this part takes about 5 sec
tic;
for i = 1:n
	rect = bbox(i, 1:4);
	im = frame(rect(2):rect(2)+rect(4), rect(1):rect(1)+rect(3), :);
	im = imresize(im, [227, 227], 'bilinear');
	im = im(:,:,[3 2 1]) - IMAGE_MEAN;
	images(:,:,:,i) = permute(im, [2 1 3]);
end
toc;

epoch = ceil(n / caffe_batch_size);
confidence = zeros(21, n);

% this part takes about 2.7 sec
tic;
for e = 1:epoch
	start = (e-1)*caffe_batch_size + 1;
	if e == epoch
		last = n;
	else
		last = start + caffe_batch_size - 1;
	end
	input_data = {images(:,:,:,start:last)};
	scores = caffe('forward', input_data);
	confidence(:, start:last) = reshape(scores{1}, [21, caffe_batch_size]);
end
toc;

confidence = confidence(16,:);

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

if max(confidence) < opt.updateThres
    param.update = true;
else
    param.update = false;
end
%}

disp(max(confidence));
confidence = confidence - min(confidence);
param.conf = exp(double(confidence) ./opt.condenssig)';
param.conf = param.conf ./ sum(param.conf);
[maxprob,maxidx] = max(param.conf);
if maxprob == 0 || isnan(maxprob)
    error('overflow!');
end
param.est = affparam2mat(param.param(:,maxidx));
%param.wimg = reshape(data(:,maxidx), sz);

if exist('coef', 'var')
    param.bestCoef = coef(:,maxidx);
end
