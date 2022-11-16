using Plots, LinearAlgebra, Statistics, Random, Distributions, CSV, DelimitedFiles, Interpolations
#--------------------------------------------------------------------------------------------------------------------------------------------
# Hugget (1996), a replica
#--------------------------------------------------------------------------------------------------------------------------------------------
# Carlos Rojas Quiroz
#--------------------------------------------------------------------------------------------------------------------------------------------
# 1. Build an equidistant grid of capital a ∈ (0,a_max), you can start with a value of a_max = 100, then we will check if we need to expand 
# or shrink the grid and a number of points naa = 100
#--------------------------------------------------------------------------------------------------------------------------------------------
a_min       =   0.0;
a_max       =   100.0;
naa         =   100;
a           =   LinRange(a_min,a_max,naa);
#--------------------------------------------------------------------------------------------------------------------------------------------
# 2. Discretize the income shock z and compute the transition probability matrix using the Tauchen method
#--------------------------------------------------------------------------------------------------------------------------------------------
μ_z         =   0.0;
ρ_z         =   0.9;
σ_ϵ         =   0.2;
nzz         =   11;
#--------------------------------------------------------------------------------------------------------------------------------------------
# (a) Tauchen function
#--------------------------------------------------------------------------------------------------------------------------------------------
function Tauchen(mu,rho,sigma,N,m)
    P           =   zeros(N,N);
    x_min       =   mu/(1-rho)-m*sqrt(sigma^2/(1-rho^2)); 
    x_max       =   mu/(1-rho)+m*sqrt(sigma^2/(1-rho^2));
    x           =   LinRange(x_min,x_max,N)';
    dx          =   (x_max-x_min)/(N-1);
    for x_i in 1:N
        for zp_i in 2:N-1 
            P[x_i,zp_i]         =   cdf(Normal(0,1),(x[zp_i]-mu-rho*x[x_i]+dx/2)/sigma)-cdf(Normal(0,1),(x[zp_i]-mu-rho*x[x_i]-dx/2)/sigma);
        end
        P[x_i,1]            =   cdf(Normal(0,1),(x[1]-mu-rho*x[x_i]+dx/2)/sigma);
        P[x_i,N]            =   1-cdf(Normal(0,1),(x[N]-mu-rho*x[x_i]-dx/2)/sigma);
    end
    return (x,P)
end
logZ        =   Tauchen(μ_z,ρ_z,σ_ϵ,nzz,4);
#--------------------------------------------------------------------------------------------------------------------------------------------
# (b) Normalize the labor market productivity vector such that the unconditional mean of the shock z equals 1
#--------------------------------------------------------------------------------------------------------------------------------------------
Z           =   exp.(logZ[1]);
Pz          =   logZ[2];
z           =   (Z.-mean(Z))./std(Z).+1;
#--------------------------------------------------------------------------------------------------------------------------------------------
# (c) Invariant distribution
#--------------------------------------------------------------------------------------------------------------------------------------------
function invariant_prod_dist(nzz, Phi)
    S           =   nzz;
    Phi_sd      =   ones(1,S)/S;
    diff        =   1;
    tol         =   0.0000001;
    while abs(diff) > tol
        Phi_sd1 = Phi_sd*Phi;
        diff    = (Phi_sd1-Phi_sd)[argmax(Phi_sd1-Phi_sd)];
        Phi_sd  = Phi_sd1;
    end
    return Phi_sd[1,:]
end
Φ           =   invariant_prod_dist(nzz, Pz);
#--------------------------------------------------------------------------------------------------------------------------------------------
# 3. Load the survival rate probabilities provided in the txt file (LifeTables.txt) and compute the fraction of population ψj in each 
# age group j
#--------------------------------------------------------------------------------------------------------------------------------------------
Sj          =   readdlm("LifeTables.txt", ';', Float64, '\n',header=false);
J           =   length(Sj);
n           =   0.01;
N0          =   1;
Ψ           =   zeros(J,1);
Ψ[1,1]      =   N0*(1+n);
for ii in 2:J
    Ψ[ii,1] =   Sj[ii-1,1]*Ψ[ii-1,1]/(1+n);
end
Ψ           =   Ψ./sum(Ψ);
#--------------------------------------------------------------------------------------------------------------------------------------------
# 4. The function describing the efficiency units of labor is given by
#--------------------------------------------------------------------------------------------------------------------------------------------
JR          =   41;
λ0          =   0.195;
λ1          =   0.107;
λ2          =   -0.00213;
e           =   zeros(J,nzz)
for jj in 1:JR-1
    e[jj,:] =   z.*(λ0+λ1*jj+λ2*jj^2);
end
#--------------------------------------------------------------------------------------------------------------------------------------------
# (a) Compute total labor supply L
#--------------------------------------------------------------------------------------------------------------------------------------------
L           =   0;
for jj in 1:JR-1
    L       =   Ψ[jj,1]*(reshape(Φ,1,nzz)*reshape(e[jj,:],1,nzz)').+L;
end
#--------------------------------------------------------------------------------------------------------------------------------------------
# 5. Assume the following parameters
#--------------------------------------------------------------------------------------------------------------------------------------------
# (a)
σ           =   2;
β           =   0.96;
# (b)
α           =   0.36;
δ           =   0.08;
A           =   1;
#--------------------------------------------------------------------------------------------------------------------------------------------
# 6. Set a value for the interest rate
#--------------------------------------------------------------------------------------------------------------------------------------------
rg          =   0.02;
#--------------------------------------------------------------------------------------------------------------------------------------------
# 7. For the guess of the interest rate and given L compute
#--------------------------------------------------------------------------------------------------------------------------------------------
# (a) Using firms FOCs: aggregate capital demand and wages
#--------------------------------------------------------------------------------------------------------------------------------------------
K           =   ((rg+δ)./α).^(1/(α-1))*L;
w           =   (1-α).*(K./L).^α;
#--------------------------------------------------------------------------------------------------------------------------------------------
# (b) Equilibrium payroll tax θ and associated pension b
#--------------------------------------------------------------------------------------------------------------------------------------------
ω           =   0.5;
θ           =   ω*sum(Ψ[JR:J,1])/sum(Ψ[1:JR-1,1]);
b_aux       =   θ*ω.*L./sum(Ψ[JR:J,1]);
b           =   zeros(J);
b[JR:J]     =   ones(J-JR+1)*b_aux;  
#--------------------------------------------------------------------------------------------------------------------------------------------
# 8. Make a guess of accidental bequests 
#--------------------------------------------------------------------------------------------------------------------------------------------
Tg          =   1.2;
#--------------------------------------------------------------------------------------------------------------------------------------------
# 9. Given all parameters, prices and transfers you can solve for the household problem to obtain the policy function gja(z, a) ∀ j ∈ [0, J]
#--------------------------------------------------------------------------------------------------------------------------------------------
# (a) Non-financial income for individual of age j with shock z
#--------------------------------------------------------------------------------------------------------------------------------------------
d           =   zeros(J,nzz);
d           =   (1-θ)*e.*w.+b.+Tg;
#--------------------------------------------------------------------------------------------------------------------------------------------
# (b) Utiliy function
#--------------------------------------------------------------------------------------------------------------------------------------------
function utility(c_aux,σ)
    naa=length(c_aux)
        for jj in 1:naa
            c_aux[jj]           =   max(c_aux[jj],0);
        end
        u           =   c_aux.^(1-σ)./(1-σ);
    return u
end
#--------------------------------------------------------------------------------------------------------------------------------------------
# (c) Backward induction
#--------------------------------------------------------------------------------------------------------------------------------------------
V_j         =   zeros(naa,nzz,J);
g_a_j       =   zeros(naa,nzz,J);
g_c_j       =   zeros(naa,nzz,J);
#   In period j=J
for kk in 1:nzz # productivity
    for ll in 1:naa # current assets 
        g_c_j[ll,kk,J]          =   max(d[J,kk].+(1+rg)*a[ll],0);
        V_j[ll,kk,J]            =   (g_c_j[ll,kk,J]).^(1-σ)./(1-σ);
    end
end
g_a_j[:,:,J]            =   zeros(naa,nzz);
#   In period j<J
for jj in 1:J-1 # age
    for kk in 1:nzz # productivity
        for ll in 1:naa # current assets
            c_aux           =   d[J-jj,kk].+(1+rg)*a[ll].-a;
            auxvar          =   utility(c_aux,σ)+(β.*(1+rg).*reshape(Pz[kk,:],1,nzz)*V_j[:,:,J-jj+1]')';
            V_j[ll,kk,J-jj]             =   maximum(auxvar);
            g_a_j[ll,kk,J-jj]           =   a[argmax(auxvar)[1]];
            g_c_j[ll,kk,J-jj]           =   max(d[J-jj,kk].+(1+rg)*a[ll]-g_a_j[ll,kk,J-jj],0);
        end
    end
end
#--------------------------------------------------------------------------------------------------------------------------------------------
# The report to be presented on the 17th of November should provide:
#--------------------------------------------------------------------------------------------------------------------------------------------
# (a) Plot of policy function a′ − a against a for two different ages (before and after retirement)
#--------------------------------------------------------------------------------------------------------------------------------------------
# Low productivity
scatter(a,g_a_j[:,3,21],label = "46 years old", color = :orange, marker = (2, 0.5))
plot!(a,a, label = "45° line", color = :black, linestyle = :dash, linealpha = 0.5, linewidth = 2)
scatter!(a,g_a_j[:,3,61],label = "86 years old", color = :green, marker = (2, 0.5))
plot!(xlabel = "current assets", ylabel = "next period assets", grid = true, legend=:bottomright)
savefig("PolicyFunction_zlow.png")
# Mid productivity
scatter(a,g_a_j[:,6,21],label = "46 years old", color = :orange, marker = (2, 0.5))
plot!(a,a, label = "45° line", color = :black, linestyle = :dash, linealpha = 0.5, linewidth = 2)
scatter!(a,g_a_j[:,6,61],label = "86 years old", color = :green, marker = (2, 0.5))
plot!(xlabel = "current assets", ylabel = "next period assets", grid = true, legend=:bottomright)
savefig("PolicyFunction_zmid.png")
# High productivity
scatter(a,g_a_j[:,9,21],label = "46 years old", color = :orange, marker = (2, 0.5))
plot!(a,a, label = "45° line", color = :black, linestyle = :dash, linealpha = 0.5, linewidth = 2)
scatter!(a,g_a_j[:,9,61],label = "86 years old", color = :green, marker = (2, 0.5))
plot!(xlabel = "current assets", ylabel = "next period assets", grid = true, legend=:bottomright)
savefig("PolicyFunction_zhigh.png")
#--------------------------------------------------------------------------------------------------------------------------------------------
# (b) Plot the euler equation error given the linear approximation of the policy function for a very fine grid (10,000 points)
#--------------------------------------------------------------------------------------------------------------------------------------------
naa_intp    =   10000;
a_intp      =   LinRange(a_min,a_max,naa_intp);
error       =   zeros(naa_intp,nzz,J);

for kk in 1:nzz # productivity
    for ll in 1:naa_intp # current assets
        g_a_intp            =   linear_interpolation(a, g_a_j[:,kk,J]);
        error[ll,kk,J]      =   (d[J,kk].+(1+rg)*a_intp[ll].-g_a_intp(a_intp[ll]))^(-σ);
    end
end

for jj in 1:J-1 # age
    for kk in 1:nzz # productivity
        for ll in 1:naa_intp # current assets
            g_a_intp_1          =   linear_interpolation(a, g_a_j[:,kk,J-jj+1]);
            g_a_intp            =   linear_interpolation(a, g_a_j[:,kk,J-jj]);
            error[ll,kk,J-jj]   =   abs((d[J-jj,kk].+(1+rg).*a_intp[ll].-g_a_intp(a_intp[ll])).^(-σ).-
            Sj[J-jj]*β*(1+rg)*Pz[kk,:]'*(d[J-jj+1,:].+(1+rg)*g_a_intp(a_intp[ll]).-g_a_intp_1(g_a_intp(a_intp[ll]))).^(-σ));
        end
    end
end
# Low productivity
plot(a_intp,error[:,3,21],label = "46 years old", color = :orange)
plot!(a_intp,error[:,3,61],label = "86 years old", color = :green)
plot!(xlabel = "current assets", ylabel = "Abs(Euler equation error)", grid = true, legend=:topright)
savefig("Euler_zlow.png")
# Mid productivity
plot(a_intp,error[:,6,21],label = "46 years old", color = :orange)
plot!(a_intp,error[:,6,61],label = "86 years old", color = :green)
plot!(xlabel = "current assets", ylabel = "Abs(Euler equation error)", grid = true, legend=:topright)
savefig("Euler_zmid.png")
# High productivity
plot(a_intp,error[:,9,21],label = "46 years old", color = :orange)
plot!(a_intp,error[:,9,61],label = "86 years old", color = :green)
plot!(xlabel = "current assets", ylabel = "Abs(Euler equation error)", grid = true, legend=:topright)
savefig("Euler_zhigh.png")