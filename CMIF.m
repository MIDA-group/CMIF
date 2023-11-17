function [C,N]=CMIF(A,B,Ma,Mb)
%function [C,N]=CMIF(A,B[,maskA,maskB])
%
% Cross Mutual Information Function for two integer valued 2D images A and B
% Recommended is to quantize the images first, e.g., using imsegkmeans
%
%   Example
%   -------
%
%   % Load images
%   onion   = imread('onion.png');
%   peppers = rgb2gray(imread('peppers.png'));
%   imshowpair(peppers,onion,'montage')
%
%   % Quantize to levels
%   Q_onion = imsegkmeans(onion,6);
%   Q_peppers = imsegkmeans(peppers,6);
%   c = CMIF(Q_peppers,Q_onion);
%   figure, surf(c), shading flat
% 
%   [ypeak, xpeak] = find(c==max(c(:)));
%   % Compute translation from max location in correlation matrix
%   yoffSet = ypeak-size(onion,1);
%   xoffSet = xpeak-size(onion,2);
%   
%   % Display matched area
%   figure
%   hAx  = axes;
%   imshow(peppers,'Parent', hAx);
%   drawrectangle(hAx, 'Position', [xoffSet+1, yoffSet+1, size(onion,2), size(onion,1)]);
%
%
% Note: CMIF uses the same convention as xcorr2, where the second argument is sliding
%  This different from normxcorr2, for which the first argument is sliding.
%
%
% [1] J. Öfverstedt, J. Lindblad, and N. Sladoje.
% Fast computation of mutual information in the frequency domain
% with applications to global multimodal image alignment.
% Pattern Recognition Letters, Vol. 159, pp. 196-203, 2022.
% https://doi.org/10.1016/j.patrec.2022.05.022
%
%Author: Joakim Lindblad

% Copyright (c) 2023, Joakim Lindblad
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Output size
osiz=size(A)+size(B)-[1,1];

% Masks
if nargin<3 || isempty(Ma), Ma=true(size(A)); end
if nargin<4 || isempty(Mb), Mb=true(size(B)); end

% Split into level sets
[lA,m]=levels(A,Ma);
[lB,n]=levels(B,Mb);
%fprintf('%d x %d levels\n',m,n);


% CC of masks
[fMa,fMb]=mkfft(Ma,Mb,osiz);
N=conv2fft(fMa,fMb);

% CC of level sets
[flA,flB]=mkfft(lA,lB,osiz);
cA=conv2fft(flA,fMb);
cB=conv2fft(fMa,flB);


% Marginal entropies
HA=Ent(cA./N);
HB=Ent(cB./N);

% CC and Entropy all combinations of m x n level sets
% Looping, since expanding all is probably too memory hungry
HAB=zeros(osiz);
for i=1:m %loop over levels in A
	caB=conv2fft(flA(:,:,i),flB); %for one level a and all levels B
	HAB=HAB+Ent(caB./N); % Accumulate joint entropies
end


% Final CMIF
C = HA+HB-HAB;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Entropy of stack(s) of probability images, summing over all but 1st two dims
function H=Ent(I)
siz=size(I,[1,2]); %Output size
[idx,~,p]=find(reshape(I,prod(siz),[])); %Bundle x,y
[i,j]=ind2sub(siz,idx);
H=accumarray([i,j],-p.*log2(p),siz);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Make FFT for left and right part of cross correlation
function [f,g]=mkfft(x,y,osiz)
f=fft2(x,osiz(1),osiz(2)); 
g=fft2(rot90(conj(y),2),osiz(1),osiz(2)); % Right part flipped (matching Matlab xcorr2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute conv2 from FFT data; convolution theorem
% Knowing that the output is integer only, rounding the output avoids floating point errors
function c=conv2fft(a,b)
c=round(abs(ifft2(a.*b,'symmetric'))); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute level set representation of a discrete valued image A (with optional ROI mask)
%
%  L is a logical array of size [size(A),n]
%  l is a vector of length n containing the levels of A
function [L,n,l]=levels(A,mask)
if nargin<2, mask=true(size(A)); end
N=numel(A);
[l,~,C]=unique(A(mask));
n=numel(l);
L=false([size(A),n]);
L(sub2ind([N,n],find(mask),C))=true;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                               END OF FILE                               %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
