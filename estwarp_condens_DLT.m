function param = estwarp_condens_DLT(frame, param, opt)

global caffe_batch_size;
global object_class;

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
	images(:,:,:,i) = imresize(frame(bbox(i,2):bbox(i,2)+bbox(i,4), bbox(i,1):bbox(i,1)+bbox(i,3), :), [227, 227]);
end
images = bsxfun( @minus, images(:,:,[3 2 1],:), IMAGE_MEAN );
images = permute(images, [2 1 3 4]);
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

confidence = confidence(object_class,:);

disp(max(confidence));
confidence = confidence - min(confidence);
param.conf = exp(double(confidence) ./opt.condenssig)';
param.conf = param.conf ./ sum(param.conf);
[maxprob,maxidx] = max(param.conf);
if maxprob == 0 || isnan(maxprob)
    error('overflow!');
end
param.est = affparam2mat(param.param(:,maxidx));

if exist('coef', 'var')
    param.bestCoef = coef(:,maxidx);
end
