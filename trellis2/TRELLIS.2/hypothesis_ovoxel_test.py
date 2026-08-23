import torch  
import torch.nn.functional as F  
from o_voxel.convert import flexible_dual_grid_to_mesh  
from trellis2.representations import Mesh  
from trellis2.renderers import MeshRenderer  
from trellis2.utils.loss_utils import l1_loss, ssim, lpips  
  
class OVoxelOptimizer:  
    def __init__(self, resolution=512, device='cuda'):  
        self.resolution = resolution  
        self.device = device  
        self.renderer = MeshRenderer({'near': 1, 'far': 3, 'resolution': 1024}, device=device)  
          
        # Loss weights (from ShapeVaeTrainer)  
        self.lambda_intersected = 0.1  
        self.lambda_vertice = 1e-2  
        self.lambda_mask = 1  
        self.lambda_depth = 10  
        self.lambda_normal = 1  
        self.lambda_ssim = 0.2  
        self.lambda_lpips = 0.2  
          
    def extract_mesh(self, coords, dual_vertices, intersected_flag, quad_lerp, train=True):  
        """Extract mesh from O-Voxel features"""  
        return flexible_dual_grid_to_mesh(  
            coords=coords,  
            dual_vertices=dual_vertices,  
            intersected_flag=intersected_flag,  
            split_weight=quad_lerp if train else None,  
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],  
            grid_size=self.resolution,  
            train=train  
        )  
      
    def compute_direct_losses(self, pred_vertices, pred_intersected, gt_vertices, gt_intersected):  
        """Direct regression losses on O-Voxel features"""  
        terms = {}  
          
        # Intersected flag BCE loss  
        if self.lambda_intersected > 0:  
            terms["direct/intersected"] = F.binary_cross_entropy_with_logits(  
                pred_intersected.flatten(), gt_intersected.flatten().float()  
            )  
          
        # Vertex MSE loss  
        if self.lambda_vertice > 0:  
            terms["direct/vertice"] = F.mse_loss(pred_vertices, gt_vertices)  
          
        return terms  
      
    def compute_rendering_losses(self, pred_mesh, gt_mesh, cameras):  
        """Rendering losses on extracted meshes"""  
        terms = {}  
          
        # Render both meshes  
        gt_renders = self._render_single(gt_mesh, cameras)  
        pred_renders = self._render_single(pred_mesh, cameras)  
          
        # Mask loss  
        terms['render/mask'] = l1_loss(pred_renders['mask'], gt_renders['mask'])  
          
        # Depth loss  
        terms['render/depth'] = l1_loss(pred_renders['depth'], gt_renders['depth'])  
          
        # Normal losses  
        terms['render/normal/l1'] = l1_loss(pred_renders['normal'], gt_renders['normal'])  
        terms['render/normal/ssim'] = 1 - ssim(pred_renders['normal'], gt_renders['normal'])  
        terms['render/normal/lpips'] = lpips(pred_renders['normal'], gt_renders['normal'])  
          
        return terms  
      
    def _render_single(self, mesh, cameras):  
        """Render a single mesh"""  
        self.renderer.rendering_options['near'] = cameras['near']  
        self.renderer.rendering_options['far'] = cameras['near'] + 2  
        return self.renderer.render(  
            mesh,   
            cameras['extrinsics'],   
            cameras['intrinsics'],   
            return_types=['mask', 'normal', 'depth']  
        )  
      
    def optimization_step(self, pred_coords, pred_dual_vertices, pred_intersected_logits,   
                          pred_quad_lerp, gt_coords, gt_dual_vertices, gt_intersected,   
                          optimizer, cameras):  
        """  
        Single optimization step  
          
        Args:  
            pred_coords: (N, 3) predicted voxel coordinates  
            pred_dual_vertices: (N, 3) predicted dual vertices  
            pred_intersected_logits: (N, 3) predicted intersected flags (logits)  
            pred_quad_lerp: (N, 1) predicted quad split weights  
            gt_coords: (N, 3) ground truth voxel coordinates  
            gt_dual_vertices: (N, 3) ground truth dual vertices  
            gt_intersected: (N, 3) ground truth intersected flags (boolean)  
            optimizer: torch optimizer  
            cameras: camera parameters dict  
        """  
        optimizer.zero_grad()  
          
        # Threshold logits to boolean for mesh extraction  
        pred_intersected_flag = pred_intersected_logits > 0  
          
        # Extract meshes  
        gt_vertices, gt_faces = self.extract_mesh(  
            gt_coords, gt_dual_vertices, gt_intersected, None, train=False  
        )  
        pred_vertices, pred_faces = self.extract_mesh(  
            pred_coords, pred_dual_vertices, pred_intersected_flag, pred_quad_lerp, train=True  
        )  
          
        # Create Mesh objects  
        gt_mesh = Mesh(gt_vertices, gt_faces)  
        pred_mesh = Mesh(pred_vertices, pred_faces)  
          
        # Compute direct losses  
        direct_terms = self.compute_direct_losses(  
            pred_dual_vertices, pred_intersected_logits, gt_dual_vertices, gt_intersected  
        )  
          
        # Compute rendering losses  
        render_terms = self.compute_rendering_losses(pred_mesh, gt_mesh, cameras)  
          
        # Combine losses  
        total_loss = 0.0  
        for key, value in direct_terms.items():  
            if 'intersected' in key:  
                total_loss += self.lambda_intersected * value  
            elif 'vertice' in key:  
                total_loss += self.lambda_vertice * value  
          
        for key, value in render_terms.items():  
            if 'mask' in key:  
                total_loss += self.lambda_mask * value  
            elif 'depth' in key:  
                total_loss += self.lambda_depth * value  
            elif 'normal/l1' in key:  
                total_loss += self.lambda_normal * value  
            elif 'normal/ssim' in key:  
                total_loss += self.lambda_normal * self.lambda_ssim * value  
            elif 'normal/lpips' in key:  
                total_loss += self.lambda_normal * self.lambda_lpips * value  
          
        # Backward pass  
        total_loss.backward()  
        optimizer.step()  
          
        return {  
            'total_loss': total_loss.item(),  
            **{k: v.item() for k, v in direct_terms.items()},  
            **{k: v.item() for k, v in render_terms.items()}  
        }
        

def optimize():
    optimizer = OVoxelOptimizer(resolution=512, device='cuda')  
  
    # Your O-Voxel features (make them trainable)  
    pred_coords = torch.randn(N, 3, requires_grad=False, device='cuda')  # Usually fixed  
    pred_dual_vertices = torch.randn(N, 3, requires_grad=True, device='cuda')  
    pred_intersected_logits = torch.randn(N, 3, requires_grad=True, device='cuda')  
    pred_quad_lerp = torch.rand(N, 1, requires_grad=True, device='cuda')  
    
    # Ground truth  
    gt_coords = ...  # From your dataset  
    gt_dual_vertices = ...  # From your dataset  
    gt_intersected = ...  # From your dataset (boolean)  
    
    # Optimizer for trainable parameters  
    opt = torch.optim.Adam([pred_dual_vertices, pred_intersected_logits, pred_quad_lerp], lr=1e-3)  
    
    # Camera parameters (simplified)  
    cameras = {  
        'extrinsics': ...,  
        'intrinsics': ...,  
        'near': 1.0  
    }  
    
    # Optimization loop  
    for step in range(1000):  
        losses = optimizer.optimization_step(  
            pred_coords, pred_dual_vertices, pred_intersected_logits, pred_quad_lerp,  
            gt_coords, gt_dual_vertices, gt_intersected,  
            opt, cameras  
        )  
        
        if step % 100 == 0:  
            print(f"Step {step}, Loss: {losses['total_loss']:.4f}")