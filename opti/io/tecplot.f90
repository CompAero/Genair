subroutine open_file_tecplot(FName)
  implicit none
  character(len=*) :: FName
  character(len=*), parameter :: &
       Title = 'GENAIR'//char(0), &
       Variables = 'X Y Z RHO RHO-U RHO-V RHO-W E'//char(0), &
       ScratchDir = '/tmp'//char(0)
  integer(kind=4) :: Filetype, Debug, VIsDouble, TECINI112, I
  Filetype = 0  !-- 0=Full, 1=Grid, 2=Solution
  Debug = 1     !-- 0=no debugging, 1=debugging
  VIsDouble = 1 !-- 0=Single, 1=Double
  I = TECINI112(Title, &
                Variables, &
                FName, &
                ScratchDir, &
                Filetype, &
                Debug, &
                VIsDouble)
end subroutine

subroutine write_zone_header(ZoneTitle, NumFaceConnections, &
                             IMax, JMax, KMax)
  implicit none
  character(len=*) :: ZoneTitle
  integer(kind=4) :: ZoneType
  integer(kind=4) :: IMax, JMax, KMax
  integer(kind=4) :: ICellMax, JCellMax, KCellMax
  real(kind=8) :: SolutionTime
  integer(kind=4) :: StrandID, ParentZone, IsBlock, NumFaceConnections, &
                     FaceNeighborMode, TotalNumFaceNodes, &
                     NumConnectedBoundaryFaces, &
                     TotalNumBoundaryConnections, &
                     ShareConnectivityFromZone, TECZNE112, I
  integer(kind=4), pointer :: PassiveVarList, ValueLocation, &
                              ShareVarFromZone
  ZoneType = 0 !-- 0=Ordered
  ICellMax = 0
  JCellMax = 0
  KCellMax = 0
  SolutionTime = 0.d0
  StrandID = 0
  ParentZone = 0
  IsBlock = 1 !-- 1=Block format
  FaceNeighborMode = 2 !-- 2=GlobalOnetoOne
  TotalNumFaceNodes = 0
  NumConnectedBoundaryFaces = 0
  TotalNumBoundaryConnections = 0
  PassiveVarList => null()
  ValueLocation => null()
  ShareVarFromZone => null()
  ShareConnectivityFromZone = 0
  I = TECZNE112(ZoneTitle, &
                ZoneType, &
                IMax, &
                JMax, &
                KMax, &
                ICellMax, &
                JCellMax, &
                KCellMax, &
                SolutionTime, &
                StrandID, &
                ParentZone, &
                IsBlock, &
                NumFaceConnections, &
                FaceNeighborMode, &
                TotalNumFaceNodes, &
                NumConnectedBoundaryFaces, &
                TotalNumBoundaryConnections, &
                PassiveVarList, &
                ValueLocation, &
                ShareVarFromZone, &
                ShareConnectivityFromZone)
end subroutine

subroutine write_zone_data(N, Dat)
  implicit none
  integer(kind=4) :: N, IsDouble, TECDAT112, I
  real(kind=8) :: Dat(:)
  IsDouble = 1 !-- 0=Single, 1=Double
  I = TECDAT112(N, &
                Dat, &
                IsDouble)
end subroutine

subroutine write_face_connections(FaceConnections)
  implicit none
  integer(kind=4) :: FaceConnections(:), TECFACE112, I
  I = TECFACE112(FaceConnections)
end subroutine

subroutine close_file_tecplot()
  implicit none
  integer(kind=4) :: TECEND112, I
  I = TECEND112()
end subroutine
