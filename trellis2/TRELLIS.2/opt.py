shape_slat_feats = torch.nn.Parameter(shape_slat.feats.clone())  
tex_slat_feats = torch.nn.Parameter(tex_slat.feats.clone())  
  
optimizer = torch.optim.Adam([  
    {'params': [shape_slat_feats], 'lr': 0.01},  
    {'params': [tex_slat_feats], 'lr': 0.01}  
])  
  
for i in range(num_iterations):  
    optimizer.zero_grad()  
      
    shape_slat_opt = shape_slat.replace(feats=shape_slat_feats)  
    tex_slat_opt = tex_slat.replace(feats=tex_slat_feats)  
      
    meshes, subs = pipeline.decode_shape_slat(shape_slat_opt, resolution)  
    tex_voxels = pipeline.decode_tex_slat(tex_slat_opt, subs)  
      
    mesh = meshes[0]  
    mesh.fill_holes()  
    mesh_with_voxel = MeshWithVoxel(  
        mesh.vertices, mesh.faces,  
        origin=[-0.5, -0.5, -0.5],  
        voxel_size=1/resolution,  
        coords=tex_voxels.coords[:, 1:],  
        attrs=tex_voxels.feats,  
        voxel_shape=torch.Size([*tex_voxels.shape, *tex_voxels.spatial_shape]),  
        layout=pipeline.pbr_attr_layout  
    )  
      
    loss = compute_loss(mesh_with_voxel)  
    loss.backward()  
    optimizer.step()
