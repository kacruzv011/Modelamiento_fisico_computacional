program benchmark
    implicit none
    integer :: i, steps
    integer(8) :: j, iteraciones
    real(8) :: p, p0, b, r, k, h
    real(8) :: start_time, end_time, total_time

    ! Parámetros
    p0 = 0.01d0
    b = 0.02d0
    r = 0.1d0
    k = r * b
    h = 1.0d0
    steps = 50
    
    ! 10 millones de iteraciones
    iteraciones = 10000000
    
    print *, "Iniciando Benchmark Fortran (", iteraciones, " iteraciones)..."

    ! Iniciar cronómetro CPU
    call cpu_time(start_time)

    do j = 1, iteraciones
        p = p0
        do i = 1, steps
            ! Método de Euler
            p = p + h * k * (1.0d0 - p)
        end do
    end do

    ! Parar cronómetro
    call cpu_time(end_time)
    total_time = end_time - start_time

    print *, "--------------------------------------"
    print *, "Valor p(50) final: ", p
    print *, "Tiempo Total Fortran: ", total_time, " segundos"
    print *, "--------------------------------------"

end program benchmark