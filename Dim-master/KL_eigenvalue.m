% Using Newton-Raphson method to solve eqation
%          (w^2-c^2)tan(wT)-2cw=0
% function [w,lamda]=KL_eigenvalue(c,T,N,tol)

function [w,lamda]=KL_eigenvalue(c,T,N,tol)  %c & T are the constants and N is the 
                                     %number of roots and tol is the error tolerance.

format long e;

MaxN=100;% the maximum number of iteration;
tp=c*T;
check=0;
ca=0;
if tp<pi/2
    CriN=0;
    check=1;
    ca=1;
else
    tp=(c*T-pi/2)/pi;
    tpn=floor(tp);
    if floor(tp)==ceil(tp)
        CriN=tpn;
        check=1;
        ca=2;
    end
end
if ~check
    CriN=tpn+1;
    ca=3;
end  % the statements above are used to look for the critical N for c;

if CriN>N
    w=zeros(N,1);
    for i=1:N
        w0=(i-0.25)*pi/T;
        tt=fw(w0,c,T);
        while tt>0
            w0=((i-0.5)*pi/T+w0)/2;
            tt=fw(w0,c,T);
        end
        NN=0;
        tolerance=1;
        while tolerance>tol & NN<MaxN
            w1=w0-fw(w0,c,T)/dfw(w0,c,T);
            tolerance=abs(fw(w1,c,T));
            w0=w1;
            NN=NN+1;
        end
        w(i)=w1;
    end
else
    switch ca
    case 1
        w=zeros(N,1);
        w0=(c+0.5*pi/T)/2;
        tt=fw(w0,c,T);
        while tt<0
            w0=(0.5*pi/T+w0)/2;
            tt=fw(w0,c,T);
        end
        NN=0;
        tolerance=1;
        while tolerance>tol & NN<MaxN
            w1=w0-fw(w0,c,T)/dfw(w0,c,T);
            tolerance=abs(fw(w1,c,T));
            w0=w1;
            NN=NN+1;
        end
        w(1)=w1;
        for i=1:N-1
            w0=(i+0.25)*pi/T;
            tt=fw(w0,c,T);
            while tt<0
                w0=((i+0.5)*pi/T+w0)/2;
                tt=fw(w0,c,T);
            end
            NN=0;
            tolerance=1;
            while tolerance>tol & NN<MaxN
                w1=w0-fw(w0,c,T)/dfw(w0,c,T);
                tolerance=abs(fw(w1,c,T));
                w0=w1;
                NN=NN+1;
            end
            w(i+1)=w1;
        end
    case 2
        w=zeros(N,1);
        if CriN>0
            for i=1:CriN
                w0=(i-0.25)*pi/T;
                tt=fw(w0,c,T);
                while tt>0
                    w0=((i-0.5)*pi/T+w0)/2;
                    tt=fw(w0,c,T);
                end
                NN=0;
                tolerance=1;
                while tolerance>tol & NN<MaxN
                    w1=w0-fw(w0,c,T)/dfw(w0,c,T);
                    tolerance=abs(fw(w1,c,T));
                    w0=w1;
                    NN=NN+1;
                end
                w(i)=w1;
            end
            for i=CriN:N
                w0=(i+0.25)*pi/T;
                tt=fw(w0,c,T);
                while tt<0
                    w0=((i+0.5)*pi/T+w0)/2;
                     tt=fw(w0,c,T);
                end
                NN=0;
                tolerance=1;
                while tolerance>tol & NN<MaxN
                    w1=w0-fw(w0,c,T)/dfw(w0,c,T);
                    tolerance=abs(fw(w1,c,T));
                    w0=w1;
                    NN=NN+1;
                end
                w(i)=w1;
            end
        else
            for i=1:N
                w0=(i+0.25)*pi/T;
                tt=fw(w0,c,T);
                while tt<0
                    w0=((i+0.5)*pi/T+w0)/2;
                    tt=fw(w0,c,T);
                end
                NN=0;
                tolerance=1;
                while tolerance>tol & NN<MaxN
                    w1=w0-fw(w0,c,T)/dfw(w0,c,T);
                    tolerance=abs(fw(w1,c,T));
                    w0=w1;
                    NN=NN+1;
                end
                w(i)=w1;
            end 
        end
    case 3
        w=zeros(N+1,1);
        for i=1:CriN-1
            w0=(i-0.25)*pi/T;
            tt=fw(w0,c,T);
            while tt>0
                w0=((i-0.5)*pi/T+w0)/2;
                tt=fw(w0,c,T);
            end
            NN=0;
            tolerance=1;
            while tolerance>tol & NN<MaxN
                w1=w0-fw(w0,c,T)/dfw(w0,c,T);
                tolerance=abs(fw(w1,c,T));
                w0=w1;
                NN=NN+1;
            end
            w(i)=w1;
         end
         w0=0.5*((CriN-0.5)*pi/T+c);
         tt=fw(w0,c,T);
         while tt>0
             w0=((CriN-0.5)*pi/T+w0)/2;
             tt=fw(w0,c,T);
         end
         NN=0;
         tolerance=1;
         while tolerance>tol & NN<MaxN
             w1=w0-fw(w0,c,T)/dfw(w0,c,T);
             tolerance=abs(fw(w1,c,T));
             w0=w1;
             NN=NN+1;
         end
         w(CriN)=w1;
         w0=0.5*((CriN+0.5)*pi/T+c);
         tt=fw(w0,c,T);
         while tt<0
             w0=((CriN+0.5)*pi/T+w0)/2;
             tt=fw(w0,c,T);
         end
         
         NN=0;
         tolerance=1;
         while tolerance>tol & NN<MaxN
             w1=w0-fw(w0,c,T)/dfw(w0,c,T);
             tolerance=abs(fw(w1,c,T));
             w0=w1;
             NN=NN+1;
         end
         w(CriN+1)=w1;
         for i=CriN+1:N
             w0=(i+0.25)*pi/T;
             tt=fw(w0,c,T);
             while tt<0
                 w0=((i+0.5)*pi/T+w0)/2;
                 tt=fw(w0,c,T);
             end
             NN=0;
             tolerance=1;
             while tolerance>tol & NN<MaxN
                 w1=w0-fw(w0,c,T)/dfw(w0,c,T);
                 tolerance=abs(fw(w1,c,T));
                 w0=w1;
                 NN=NN+1;
             end
             w(i+1)=w1;
          end  
      end
  end % above is Newton method;
  
  lamda=zeros(length(w),1);
  lamda=(2*c)./(w.^2+c^2);
  