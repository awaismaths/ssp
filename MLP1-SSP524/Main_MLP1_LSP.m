
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Mass-lumped P1 code for u_t-Delta zeta(u)=f with Dirichlet BC zeta(u)=zb
% Time decretization: u^(n)-Delta(zeta(u^(n))=u^(n-1)+dt*f^(n)=:b
% To linearise, zeta(u^(n) is replaced by
% L(u^(n,i)-u^(n,i-1))+zeta(u^(n,i-1)), i.e.,
%  u^(n,i)+dt*L*Delta u^(n,i)=dt Delta (zeta(u^(n,i-1))- L u^(n,i-1))+b
% For unknown X=u^(n,i), weak form is: GX:=(M+dt*L*A)X=-dt*A (zeta(Xprev)-L*Xprev)+rhs=:B
% BC are replaced with: L(X_b-Xprev_b))+zeta(Xprev_b)=zeta(exact_sol_b)
clear all;

%  function [nts,ITER,Time]=Main_MLP1_LSP(dt_initial,meshes)
format long;

% Parameters for nonlinearity and barenblatt solution
% global CB;
%global ucase zcase;
ucase=2;
zcase=2;


%t0=.1;
% CB=0.005;
% Lc>L_z/2 where L_z is Lipschitz constant
Lc=0.65;
% Final times
T=1;
 %dt_initial=0.1;
% nonlinear iterations
%tol=1e-5;
itermax=50000;
% Relaxation parameter (1 for no relaxation)
relax = 1;

%%
% Sequence of meshes over which we want to run the scheme
%%
% The meshes are available at https://github.com/jdroniou/HHO-Lapl-OM
meshes={'mesh1_1.mat'};%'mesh1_2.mat';'mesh1_3.mat'};%'mesh1_4.mat'};%'mesh1_05.mat'};%'mesh1_6.mat'};%'mesh1_7.mat'};

nbmeshes=size(meshes,1);
Lp1_error=zeros(nbmeshes,1);
L2zeta_error=zeros(nbmeshes,1);
h=zeros(nbmeshes,1);
Ndt=zeros(nbmeshes,1);

% To see the results printed in file
fid = fopen('results.txt','w');

%%%fclose(fid);
% Ndt(1) = ceil(T/dt_initial);

for imesh=1:nbmeshes
    % Load mesh
    loadmesh=strcat('load ../HHO-Lapl-OM-master/matlab_meshes/',meshes{imesh});
    str = sprintf('%s\n',loadmesh);
    forkprint(fid,str);
    eval(loadmesh);
    % Compute real centers of gravity, mesh diameter and area of dual mesh
    cg=gravity_centers(ncell,cell_v,vertex,area);
    h(imesh)=max(abs(diam));
    
     % Finding boundary vertices
    fbc=zeros(size(vertex,1),1);
    bdry_vert=zeros(size(vertex,1),2);
    for i=1:ncell
        I=find(cell_n{i}==0);
        if (size(I,2)>0)
            bdry_vert_indices = [cell_v{i}(I) cell_v{i}(I+1)];
            bdry_vert(bdry_vert_indices,:)  = vertex(bdry_vert_indices,:);
            fbc(bdry_vert_indices)=1;
        end
    end
    I_indices=find(~fbc);
    B_indices=find(fbc);

    dualarea=compute_dualarea(area,ncell,nvert,cell_v,B_indices);
    % Time steps
    %Ndt(imesh)=ceil(T/h(imesh)^2);
    Ndt(imesh)=2*round(0.5*T/h(imesh)^2);
    %if (imesh>1)
    %    Ndt(imesh) = Ndt(imesh-1); 
    %end;

    str = sprintf('mesh= %s, h= %4.2e, time step= %4.2e \n',meshes{imesh},h(imesh),T/Ndt(imesh));
    forkprint(fid,str);
  % Time steppings and tolerance (Since expected error is of order dt + h)
    dt=T/Ndt(imesh);
    tol=dt/100;
    %% Initialise RHS and unknown
    % Initial condition and exact solution
    ex_sol=exact_solution(0,vertex,ucase)'; % Exact solution at t0=0

    %     write_solution_vtk(ex_sol,strcat('VTKout/solution0'),ncell,nedge,nvert,cell_v,cell_n,cell_e,vertex);
    %     write_solution_vtk(ex_sol,'VTKout/ex_sol0',ncell,nedge,nvert,cell_v,cell_n,cell_e,vertex);

    X=ex_sol;

    %% Time starts here !!!
    tic
    %% Assemble Mass and Laplacian Matrices
    [A,b]=assemble_diffusion_system(cell_v,ncell,nvert,vertex);
    Mass = spdiags(dualarea,0,nvert,nvert);
     G=Mass+dt*Lc*A;
     L=chol(G(I_indices,I_indices));

    X_b=zeros(nvert,1);
    zb=zeros(nvert,1);
    ITER=0;
    Res=0;
    %%%%%%%%%%% Begin time stepping %%%%%%%%%%%%%%
    for idt=1:Ndt(imesh);
        %Source: To compute source, I need (u_t)-laplace(zeta)
        b=assemble_source(idt*dt,cell_v,ncell,nvert,area,cg,zcase,ucase);
        %Dirichlet non-homogeneous BC
        zb(B_indices)=zetau((exact_solution(idt*dt,bdry_vert(B_indices,:),ucase))',zcase);
        % Solution: non-linear iterations
        Xprev = X;
        %
        rhs = Mass*Xprev + dt*b;
        iter = 0;
        res = 1;
        while (iter < itermax && res > tol)
            B=-dt*A*(zetau(Xprev,zcase)-Lc*Xprev)+rhs;
            X_b=zeros(nvert,1);%Xprev;
            X_b(B_indices)=(1/Lc)*(zb(B_indices)-zetau(Xprev(B_indices),zcase))+Xprev(B_indices);
          
            F=B-G*X_b;
              % bicgstab
%                         [L,U] = ilu(G(I_indices,I_indices),struct('type','ilutp','droptol',1e-8));
%                         [Xsol,flag]=bicgstab(G(I_indices,I_indices),F(I_indices),1e-8,20,L,U);
%                         if (flag ~= 0)
%                             flag
%                             error('bicgstab did not converge')
%                         end
            %
            % Using Cholesky decomposition
            Xsol=L\(L'\F(I_indices));
%           Xsol(I_indices)=G(I_indices,I_indices)\F(I_indices);
            X=X_b;
            X(I_indices)=Xsol;
             % X([B_indices;I_indices])=[X_b(B_indices);Xsol];
            iter = iter+1;
            % residual by increments
            %res = norm(X-Xprev,Inf) / norm(Xprev,Inf);     
            Xprev=X;
            % residual of non-linear system
            residue=Mass*X+dt*A*zetau(X,zcase)-rhs;
            res = norm(residue(I_indices), 2);
        end; %end non-linear iterations
        
        Res=Res+abs(res);
        ITER=ITER+iter;        
        usol=X;
        usol(B_indices)=zetau(X(B_indices),zcase);%zb(B_indices);
        if (iter==itermax)
            res
            iter
            error('no convergence')
        end;
        ex_sol=exact_solution(idt*dt,vertex,ucase)';
        % Write the solution and grid vtk files, to be plotted by "paraview"
        %         write_solution_vtk(usol,strcat('VTKout/solution',num2str(idt)),ncell,nedge,nvert,cell_v,cell_n,cell_e,vertex);

 
        %         write_solution_vtk(ex_sol,strcat('VTKout/ex_sol',num2str(idt)),ncell,nedge,nvert,cell_v,cell_n,cell_e,vertex);
        %str = sprintf('Solution computed, iter=%d, res=%4.2e, max sol=%f, max ex_sol=%f\n', iter, res, max(usol(I_indices)), max(ex_sol(I_indices)));
         %forkprint(fid,str);
        %z_error=norm(usol(B_indices)-zetau(ex_sol(B_indices),zcase),"inf")
        %res_B = norm(ex_sol(B_indices)-X(B_indices),Inf) / norm(ex_sol(B_indices),Inf)
        % res_I = norm(ex_sol(I_indices)-X(I_indices),Inf) / norm(ex_sol(I_indices),Inf)
    end % end of time stepping
    Time=toc;
    ITER=ceil(ITER/Ndt(imesh));
    Res=(Res/Ndt(imesh));
    nts=Ndt(imesh);

    str = sprintf('Mesh %i, num_time_steps=%d, Avg_iter=%d with Avg_residue=%4.2e in %f seconds\n',imesh,Ndt(imesh),ITER,Res,Time);
    forkprint(fid,str);
    clear ITER Res
           % Exact solution
                ex_sol=exact_solution(dt*idt,vertex,ucase)';
               
     % compute error
    [Lp1_error(imesh) L2zeta_error(imesh) H1zeta_error(imesh)] = compute_errors(usol,ex_sol,dualarea,A,I_indices,zcase,ucase);

    str = sprintf('Mesh %i. Errors: L^(2)=%4.2e, L^2 on zeta(u):%4.2e, H1 on zeta(u):%4.2e\n',imesh,Lp1_error(imesh),L2zeta_error(imesh),H1zeta_error(imesh));
    forkprint(fid,str);
    % nb of interior vertices
%     nvert_int(imesh) = size(I_indices,1);
   
end  %end of meshes
disp(['Elapsed time is ' num2str(toc) ' seconds' ])%round(toc/60,1)

% convergence rate
for imesh=1:nbmeshes-1
    ocLp1(imesh)=log(Lp1_error(imesh)/Lp1_error(imesh+1))/log(h(imesh)/h(imesh+1));
    ocL2zeta(imesh)=log(L2zeta_error(imesh)/L2zeta_error(imesh+1))/log(h(imesh)/h(imesh+1));
    ocH1zeta(imesh)=log(H1zeta_error(imesh)/H1zeta_error(imesh+1))/log(h(imesh)/h(imesh+1));
end

str = sprintf('Errors in L^(2) and orders of convergence:\n');
forkprint(fid,str);
for imesh=1:nbmeshes
    if (imesh==1)
        str = sprintf('\t%4.2e\n',Lp1_error(imesh));
        forkprint(fid,str);
    else
        str = sprintf('\t%4.2e \t %4.2e\n',Lp1_error(imesh),ocLp1(imesh-1));
        forkprint(fid,str);
    end
end

str = sprintf('\nErrors in L^2 on zeta(u) and orders of convergence:\n');
forkprint(fid,str);
for imesh=1:nbmeshes
    if (imesh==1)
        str = sprintf('\t%4.2e\n',L2zeta_error(imesh));
        forkprint(fid,str);
    else
        str = sprintf('\t%4.2e \t %4.2e\n',L2zeta_error(imesh),ocL2zeta(imesh-1));
        forkprint(fid,str);
    end
end

str = sprintf('\nErrors in H1 on zeta(u) and orders of convergence:\n');
forkprint(fid,str);
for imesh=1:nbmeshes
    if (imesh==1)
        str = sprintf('\t%4.2e\n',H1zeta_error(imesh));
        forkprint(fid,str);
    else
        str = sprintf('\t%4.2e \t %4.2e\n',H1zeta_error(imesh),ocH1zeta(imesh-1));
        forkprint(fid,str);
    end
end

fclose(fid);
% 
% % Write data file
% fid = fopen('data_rates_LSP.dat','w');
% fprintf(fid,'meshsize timestep L2error_zeta Lp1error H1error NvertInt Time\n');
% for i=1:nbmeshes
%     fprintf(fid,'%f %f %f %f %d %f %f\n',h(i),T/Ndt(i),L2zeta_error(i),Lp1_error(i),H1zeta_error(i),nvert_int(i),Time(i));
% end;

