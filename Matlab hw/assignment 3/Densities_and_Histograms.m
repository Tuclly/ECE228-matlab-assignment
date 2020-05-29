function [xarray,p,xhistarray,phist]=norm1d(mu,sigma,xmin,xmax,N,dx)
% function [p,phist]=norm1d(xmin,xmax,dx)
%
% Function computes the density of the 1D normal distribution
% at N points in the interval [xmin,xmax]
% and a histogram with bin-width dx in the same interval.
%
% INPUT
%
%  mu       mean value
%  sigma    variance
%  xmin     lower interval limit
%  xmax     upper interval limit
%  N        number of points in the interval
%  dx       bin width of histogram in the interval [xmin,xmax].
%
% OUTPUT
%
% p         density values N*1 array
% phist     histogram values  ceil((xmax-xmin)/dx) * 1 array

xarray          = linspace(xmin, xmax, N); %range of x values for the density
Nhist           = (xmax - xmin) / dx;%numer of points in the histogram
xhistarray      = xmin:dx:xmax;%range of x values for the histogram
                             
p               = (1/sqrt(2*pi*(sigma^2))) * exp(-1.*(xarray-mu).^2./(2*sigma^2));%normal distribution of mean mu, variance sigma at points given in xarray
phist           = (1/sqrt(2*pi*(sigma^2))) * exp(-1.*(xhistarray-mu).^2./(2*sigma^2)) .* dx;%histogram bin frequencies of normal distribution in 'p' for bin widths dx
end