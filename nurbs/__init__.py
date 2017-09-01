import basis
import conics
import curve
import fit
import iges
import knot
import point
import surface
import transform
import util
import volume


tools = (curve.make_linear_curve,
         curve.make_composite_curve,
         curve.param_to_arc_length,
         curve.arc_length_to_param,
         curve.reparam_arc_length_curve,

         surface.make_bilinear_surface,
         surface.make_general_cylinder,
         surface.make_ruled_surface,
         surface.make_general_cone,
         surface.make_revolved_surface_rat,
         surface.make_revolved_surface_nrat,
         surface.make_skinned_surface,
         surface.make_swept_surface,
         surface.make_gordon_surface,
         surface.make_coons_surface,
         surface.make_nsided_region,
         surface.make_composite_surface,

         volume.make_trilinear_volume,
         volume.make_ruled_volume,
         volume.make_ruled_volume2,
         volume.make_trilinear_interp_volume,
         volume.make_composite_volume,

         conics.make_circle_rat,
         conics.make_circle_nrat,
         conics.make_ellipse,
         conics.make_hyperbola,
         conics.make_parabola,

         fit.refit_curve,
         fit.refit_surface)

class _VirtualModule(object):
    def __init__(self, tools):
        for tool in tools:
            setattr(self, tool.__name__, tool)
tb = _VirtualModule(tools) # toolbox
