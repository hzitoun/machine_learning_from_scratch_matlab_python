t = 0:0.01:0.98;
y1 = sin(2*pi*4*t);
plot(t, y1);
y2 = cos(2*pi*4*t);
hold on % used to plot n functions in same window
plot(t, y2, 'r');
xlabel('time')
ylabel('value')
legend('sin', 'cos')
title('my plot')
print -dpng 'myPlot.png'
%see help plot
close % close figure
figure(1); plot(t, y1);
figure(2); plot(t, y2);

%divide plot into 1x2 grid, access first element
subplot(1, 2, 1);
plot(t, y1);
subplot(1, 2, 2);
plot(t, y2);

%change axis scale
axis([0.5 1 -1 1])

% clear figure
clf; 

A = magic(5);
imagesc(A), colorbar, colormap gray; % comma chaining commands




