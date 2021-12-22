!==========================================================================
  module func

  implicit none

  contains
!==========================================================================

!==========================================================================
      subroutine decomplu(a,p,n) 

      integer :: n,i,j,k,ik,im,ii,p(n)
      real(kind=8) :: a(n,n),b(n),x(n),s,amax
!     call printa(a,b,n)
      do i=1,n
         p(i) = i
      enddo
      do k=1,n
         amax = 0.0
         im = p(k)
         do i=k,n
            ii = p(i)
            do j=1,k-1
               a(ii,k) = a(ii,k) - a(ii,j) * a(p(j),k)
            enddo
            if (abs(a(ii,k)).gt.amax) then 
               amax = abs(a(ii,k))
               im = ii
               ik = i
            endif
         enddo
         p(ik) = p(k)
         p(k) = im
         do j=k+1,n
            ik = p(k)
            do i=1,k-1   
               a(ik,j) = a(ik,j)-a(ik,i)*a(p(i),j)
            enddo
            a(p(j),k) = a(p(j),k) / a(p(k),k)
         enddo
      enddo
      return
      end
!==========================================================================

!==========================================================================
      subroutine solve_lu(a,b,x,p,n)
      integer :: n,i,j,k,p(n)
      real(kind=8) :: a(n,n),x(n),b(n),det
      x(1) = b(p(1))
      do i=2,n
         x(i) = b(p(i))
         do j=1,i-1 
            x(i) = x(i) - a(p(i),j) * x(j)
         enddo
      enddo
      det = a(p(n),n)
      x(n) = x(n) / a(p(n),n)
      do i=n-1,1,-1
         do j=i+1,n
            x(i) = x(i) - a(p(i),j) * x(j)
         enddo
         det = det * a(p(i),i)
         x(i) = x(i) / a(p(i),i)
      enddo
      write(*,*) ' det PA = ', det
      end
!==========================================================================

!==========================================================================
      subroutine exchange(x,n,sgn,xmax,sgnmax)
      integer :: n,i1,i2,k
      real(kind=8) :: x(n),sgn,xmax,sgnmax
      
      if (xmax.lt.x(1)) then
         if (sgn*sgnmax.lt.0) then
            x(1) = xmax
          else
            x(1:n-1) = x(2:n)
            x(n) = xmax
         endif
         return
      endif
      if (xmax.gt.x(n)) then
         if ((-1)**n * sgn*sgnmax.gt.0) then
            x(n) = xmax
          else
            x(2:n) = x(1:n-1)
            x(1) = xmax
         endif
         return
      endif
      i1 = 1
      i2 = n
      do 
         if (i2-i1.eq.1) exit
         k = (i1+i2)/2
         if (xmax.lt.x(k)) then 
           i2 = k
          else
           i1 = k
         endif
      enddo
      if ((-1)**i1 * sgn*sgnmax.gt.0) then
         x(i1) = xmax
       else
         x(i2) = xmax
      endif
      end subroutine exchange
!==========================================================================

!==========================================================================
      subroutine set_matrix(a,b,x,nd,l,n,gama,ini)
      integer :: nd,l,n,i,j,mi,k,ini
      real(kind=8) :: a(nd,nd),b(nd),x(n),gama,pi,h

      if (ini.eq.1) then
         pi = acos(-1.0d0)
         h = pi / n
         do i=1,n
            x(i) = i*h
         enddo
      endif
      a(1,:) = 1.0d0
      b(1) = 1.0d0
      mi = -1
      do i=1,l
         a(i+1,1) = 0.0d0
         a(i+1,nd) = 0.0d0
         do j=1,nd-2
            a(i+1,j+1) = j**(2*i)
            b(i+1) = mi * gama ** (2*i)
         enddo
         mi = - mi
      enddo
      mi = -1
      do i=1,n
         k = i+l+1
         a(k,1) = 1.0d0
         a(k,nd) = mi
         do j=1,nd-2
            a(k,j+1) = cos(j*x(i))
            b(k) = cos(gama*x(i))
         enddo
         mi = - mi
      enddo
      end subroutine set_matrix
!==========================================================================

!==========================================================================
      subroutine best_app(a,b,p,x,nd,l,n,m,gama,c)
      integer ::n,m,l,nd,p(nd),k,i
      real(kind=8) :: gama,xmax,sgnmax,fmax,eps,sgn,y,h
      real(kind=8) :: a(nd,nd), b(nd), c(nd), x(n)

      eps = 1.0d-05
      k = 1
      h = acos(-1.d0)/200
      do
         fmax = abs(c(nd))
         call glob_max(gama,c,m,xmax,sgnmax)

      do i=0,200
         y = i*h
         write(10*k+7,*) y, cos(gama*y)
         write(10*k+8,*) y,  g(y,c,m+1)
         write(10*k+9,*) y, cos(gama*y)- g(y,c,m+1)
      enddo
      do i=1,n
         write(10*k+6,*) x(i), cos(gama*x(i))- g(x(i),c,m+1)
      enddo
         write(10*k+6,*) xmax, cos(gama*xmax)- g(xmax,c,m+1)
      k= k + 1
         if ((abs(error(xmax,gama,c,m))-fmax).lt.eps) exit
         sgn = fmax / c(nd)
         call exchange(x,n,sgn,xmax,sgnmax)
         write(*,*) ' error ', fmax
         write(*,*) ' xmax ',xmax
         write(*,*) ' x ',x
         call set_matrix(a,b,x,nd,l,n,gama,0)
         call decomplu(a,p,nd)
         call solve_lu(a,b,c,p,nd)
      enddo

      end subroutine best_app
!==========================================================================

!==========================================================================
      subroutine glob_max(gama,c,m,xmax,sgnmax)
      integer :: m,i,k
      real(kind=8) :: gama,c(m+1),xmax,xm,sgnmax,h,f,f1,f2,f3,fm,pi,a,b
      k = 100
      pi = acos(-1.d0)
      h = pi / k
      f1 = error(0.0d0,gama,c,m)
      f2 = error(h,gama,c,m)
      fm = f1
      xmax = 0.0d0
       write(15,*)' glob_max called '
      do i=2,k
         a = (i-2)*h
         b = i*h
         f3 = error(b,gama,c,m)
         if ((f2-f1)*(f3-f2).le.0.0d0) then
            if((f2-f1).gt.0.0d0) then
               write(15,*)' enter max_fiba',a,b 
               xm = max_fib(a,b,gama,c,m)
               write(15,*)' xm = ',xm
              elseif ((f3-f2).lt.0.0d0) then
               write(15,*)' enter max_fibb',a,b 
               xm = max_fib(a,b,gama,c,m)
               write(15,*)' xm = ',xm
              else
               write(15,*)' enter min_fib',a,b 
               xm = min_fib(a,b,gama,c,m)
               write(15,*)' xm = ',xm
            endif
            f = error(xm,gama,c,m)
            if (abs(f).gt.abs(fm)) then
               fm = f
               xmax = xm
            endif
            write(15,*)' fm = ',fm, ' f = ',f
         endif
         f1 = f2
         f2 = f3
      enddo
      if (abs(f3).gt.abs(fm)) then
          fm = f3
          xmax = pi
      endif
      sgnmax = fm / abs(fm)
      end subroutine glob_max
!==========================================================================

!==========================================================================
      function max_fib(a,b,gama,coef,m)
      integer :: m
      real(kind=8) :: a,b,c,d,r,max_fib,gama,coef(m+1),eps
      r = (sqrt(5.d0)+1.0d0)/2.0d0
      eps = 1.0d-05
      do
         c = b - (b-a)/r
         d = a + (b-a)/r
         if (abs(c-d).lt.eps) exit
         if (error(c,gama,coef,m).gt.error(d,gama,coef,m)) then
            b = d
          else
            a = c
         endif
      enddo
      max_fib = (a+b)/2.0d0
      end function max_fib
!==========================================================================

!==========================================================================
      function min_fib(a,b,gama,coef,m)
      integer :: m
      real(kind=8) :: a,b,c,d,r,min_fib,gama,coef(m+1),eps
      r = (sqrt(5.d0)+1.0d0)/2.0d0
      eps = 1.0d-05
      do
         c = b - (b-a)/r
         d = a + (b-a)/r
         if (abs(c-d).lt.eps) exit
         if (error(c,gama,coef,m).lt.error(d,gama,coef,m)) then
            b = d
          else
            a = c
         endif
      enddo
      min_fib = (a+b)/2.0d0
      end function min_fib
!==========================================================================

!==========================================================================
      function g(x,c,n)
      integer :: n,k
      real(kind=8) :: g,c(n),x
      g = c(1)
      do k=2,n
        g = g + c(k)*cos((k-1)*x)
      enddo
      end function g
!==========================================================================

!==========================================================================
      function error(x,gama,c,m)
      integer :: m   
      real(kind=8) :: gama, c(m+1), x, error
      error = cos(gama*x)- g(x,c,m+1)
      end function error
!==========================================================================

!==========================================================================            
      subroutine printa(a,b,n)
      integer :: n,i,j
      real(kind=8) :: a(n,n),b(n)
      write(*,*) 'matrix A'
      do i=1,n
         write(*,*) (a(i,j),j=1,n)
      enddo
      return
      end
!==========================================================================

!==========================================================================
end module func
!==========================================================================

!==========================================================================
program sistema

use func
implicit none

      integer :: n,i,j,m,l,nd 
      integer, allocatable :: p(:)
      real (kind=8), allocatable  :: x(:),b(:),a(:,:),c(:)
      real (kind=8) :: gama, pi, h, y
      write(*,*) 'entre  m,gama,l'
      read(*,*) m,gama,l
      n = m-l+1
      nd = m+2
      pi = acos(-1.d0)
      h = pi/200
      allocate (x(n))
      allocate (p(nd))
      allocate (b(nd))
      allocate (c(nd))
      allocate (a(nd,nd))
      call set_matrix(a,b,x,nd,l,n,gama,1)
      call decomplu(a,p,nd) 
      call solve_lu(a,b,c,p,nd)
      call best_app(a,b,p,x,nd,l,n,m,gama,c)
      write(*,*) ' solução ', c
      do i=0,200
         y = i*h
         write(7,*) y, cos(gama*y)
         write(8,*) y,  g(y,c,m+1)
         write(9,*) y, cos(gama*y)- g(y,c,m+1)
      enddo
      do i=1,n
         write(10,*) x(i), cos(gama*x(i))
         write(11,*) x(i),  g(x(i),c,m+1)
         write(12,*) x(i), cos(gama*x(i))- g(x(i),c,m+1)
      enddo
end  program sistema
!==========================================================================
