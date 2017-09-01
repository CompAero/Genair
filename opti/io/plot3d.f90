subroutine open_file(file_name, endian)
  !--  endian: native, swap, big_endian, little_endian
  character(len=128) :: file_name, endian
  integer :: file_stat
  open(unit=10, file=file_name, status='unknown', &
       form='unformatted', action='readwrite', iostat=file_stat, &
       convert=endian) !-- gfortran specific
  if (file_stat /= 0) then
    write(*,*) 'Error in open_file :: error opening ',file_name
    stop
  end if
end subroutine

subroutine close_file()
  close(10)
end subroutine

!-- read

subroutine read_nblk(nblk)
  integer, intent(out) :: nblk
  read(10) nblk
end subroutine

subroutine read_header(jkmmax, nblk)
  integer, intent(inout) :: jkmmax(:,:)
  integer :: nblk, di, ib
  read(10) ((jkmmax(di,ib), &
       di=1,3), ib=1,nblk)
end subroutine

subroutine read_param(jmax, p)
  real(kind=8), intent(inout) :: p(:)
  integer :: jmax, i
  read(10) (p(i), &
       i=1,jmax)
end subroutine

subroutine read_one_block(jkmmax, ndi, b)
  real(kind=8), intent(inout) :: b(:,:,:,:)
  integer :: jkmmax(:), ndi, j, k, m, di
  read(10) ((((b(j,k,m,di), &
       j=1,jkmmax(1)), k=1,jkmmax(2)), m=1,jkmmax(3)), di=1,ndi)
end subroutine

!-- write

subroutine write_nblk(nblk)
  integer :: nblk
  write(10) nblk
end subroutine

subroutine write_header(jkmmax, nblk)
  integer :: jkmmax(:,:), nblk, di, ib
  write(10) ((jkmmax(di,ib), &
       di=1,3), ib=1,nblk)
end subroutine

subroutine write_param(jmax, p)
  real(kind=8) :: p(:)
  integer :: jmax, i
  write(10) (p(i), &
       i=1,jmax)
end subroutine

subroutine write_one_block(jkmmax, ndi, b)
  real(kind=8) :: b(:,:,:,:)
  integer :: jkmmax(:), ndi, j, k, m, di
  write(10) ((((b(j,k,m,di), &
       j=1,jkmmax(1)), k=1,jkmmax(2)), m=1,jkmmax(3)), di=1,ndi)
end subroutine

subroutine write_header_1d(jmax, nblk)
  integer :: jmax(:), nblk, ib
  write(10) (jmax(ib), &
       ib=1,nblk)
end subroutine

subroutine write_one_block_1d(jmax, ndi, b)
  real(kind=8) :: b(:,:)
  integer :: jmax, ndi, j, di
  write(10) ((b(j,di), &
       j=1,jmax), di=1,ndi)
end subroutine
