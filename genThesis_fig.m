
idx = 5:5:150;

f1 = figure; 
plot(idx,err_an(idx),'Marker','s','MarkerSize',7,'LineWidth',2,'LineStyle','none'); hold;
plot(idx,err_net(idx),'Marker','x','MarkerSize',7,'LineWidth',2,'LineStyle','none');

legend('ARMA with warm restart','MLP (proposed)','Location','NorthWest' );
legend boxoff;
xlabel('Graph snapshot (t)')
title('L2-error')

% figure parameters
fig_par.width = 5;
fig_par.height = 3;
fig_par.alw = 1;
fig_par.fsz = 12;

export_figure(f1,'NetPerformanceL1.png',fig_par');