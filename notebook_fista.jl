### A Pluto.jl notebook ###
# v0.17.7

using Markdown
using InteractiveUtils

# ╔═╡ ca84f880-8370-11ec-2ed9-1fd70af33535
begin 
	import Pkg
    # activate the shared project environment
    Pkg.activate(Base.current_project())

	using MRIReco, Plots
	include("utils_MP2RAGE.jl");
end

# ╔═╡ c75e0cb0-6220-4dcd-b8d6-26a6f1a64f28
begin
	# Comparison to BART
	#Pkg.add(url = "https://github.com/aTrotier/BartIO.jl")
	using BartIO, PyCall
	pyhelp = pybuiltin("help")
	bart = BartIO.initBart("/Users/aurelien/Documents/SOFTWARE/bart","/Users/aurelien/Documents/SOFTWARE/bartpy")
	
	pyhelp(bart.pics) # check the terminal where pluto is launched
end

# ╔═╡ 5fd0776a-3deb-4c70-8bd3-3323876e4902
html"""<style>
/*              screen size more than:                     and  less than:                     */
@media screen and (max-width: 699px) { /* Tablet */ 
  /* Nest everything into here */
    main { /* Same as before */
        max-width: 1000px !important; /* Same as before */
        margin-right: 100px !important; /* Same as before */
    } /* Same as before*/

}

@media screen and (min-width: 700px) and (max-width: 1199px) { /* Laptop*/ 
  /* Nest everything into here */
    main { /* Same as before */
        max-width: 1000px !important; /* Same as before */
        margin-right: 200px !important; /* Same as before */
    } /* Same as before*/
}

@media screen and (min-width:1200px) and (max-width: 1920px) { /* Desktop */ 
  /* Nest everything into here */
    main { /* Same as before */
        max-width: 1000px !important; /* Same as before */
        margin-right: 300px !important; /* Same as before */
    } /* Same as before*/
}

@media screen and (min-width:1921px) { /* Stadium */ 
  /* Nest everything into here */
    main { /* Same as before */
        max-width: 1000px !important; /* Same as before */
        margin-right: 100px !important; /* Same as before */
    } /* Same as before*/
}
</style>
"""

# ╔═╡ af69fdc6-7abc-45f6-b5a9-59e5fd8deacb
md"
# Initiate specific environement
This notebook uses specific version of package :
- the master branch of MRIReco (manage multicoil bruker files) -> commit : 2978ca7
- the master branch of RegularizedLeastSquare (Fista bugfix with Linear Operator 2.0)-> commit : e22af54

"

# ╔═╡ 2903d8af-3b4b-4ad4-b480-037a37817b62
md"# Load the data
Defining global parameters"

# ╔═╡ 30e21633-d8d4-49f1-9fa6-4c922a43d222
begin
	slice = 25 # slice to show
	
	b = BrukerFile("data/LR_3T_CS4")
end

# ╔═╡ 0c47b8c5-76a4-4128-a525-b7b6f7b2c1fc
(acq,raw) = acqDataFromMP2RAGE(b); # create an object with function in utils_MP2RAGE.jl

# ╔═╡ 3fb00049-3b01-49dc-9f60-66a989a2508e
md" Plot the mask"

# ╔═╡ 92eef8c2-55d2-4db2-9967-9a381ae5440b
begin# check mask
	mask = zeros(acq.encodingSize[1],acq.encodingSize[2],acq.encodingSize[3]);
	for i =1:length(acq.subsampleIndices[1]);
	  mask[acq.subsampleIndices[1][i]]=1;
	end 
	plotmask = heatmap( mask[64,:,:,1], c=:grays, aspect_ratio = 1,legend = :none , axis=nothing)
end


# ╔═╡ 8a459674-94d3-4dd3-b3ea-7b4be4acee98
md"## Calculate sensitivities"

# ╔═╡ bb732ec5-6eff-4baf-8b6e-204cb21bf61d
begin
	kspace = extract3DKSpace(acq);
	
	calibSize = parse.(Int,b["CenterMaskSize"]);
	calibData = crop(kspace[:,:,:,2,:],(calibSize,calibSize,calibSize));
	
	kspace = Nothing
	
	sensSize = (acq.encodingSize[1],acq.encodingSize[2],acq.encodingSize[3]) # (size(kspace,1),size(kspace,2),size(kspace,3))
	@time sens_spirit = espirit(calibData,sensSize ,(6,6,6),eigThresh_2 = 0); # really long on my mac
	sens_spirit = convert.(ComplexF64,sens_spirit);
	sens_spirit = reshape(sens_spirit,acq.encodingSize[1],acq.encodingSize[2],acq.encodingSize[3],:);

	heatmap( abs.(sens_spirit[:,:,slice,1]), c=:grays, aspect_ratio = 1,legend = :none , axis=nothing)
end

# ╔═╡ 1e5213aa-c4e1-422e-8600-0c00978a975c
md"# Perform direct reconstruction with coil combinaison"

# ╔═╡ d951482f-d45c-4471-9d21-1fbae3bfb678
begin
	# direct reco
	params = Dict{Symbol, Any}()
	params[:reco] = "direct"
	params[:reconSize] = sensSize
	Ireco = reconstruction(acq, params)
	
	sens = reshape(sens_spirit,acq.encodingSize[1],acq.encodingSize[2],acq.encodingSize[3],1,4);
	Isense = sum(conj.(sens).*Ireco,dims=5) ./sum(abs.(sens).^2,dims = 5)
	
	MP2_sense = mp2rage(Isense)
end

# ╔═╡ 554f16f7-3592-43ab-b90a-d60b90075384
	heatmap( abs.(MP2_sense[:,:,slice,1,1]), c=:grays, aspect_ratio = 1,legend = :none , axis=nothing)

# ╔═╡ f20cb182-2c8b-4892-ae00-54601571a06c
md" # Wavelet reconstruction"

# ╔═╡ 50323e44-338c-48f0-a2f3-b024571638c2
begin
	# Then Wavelet
	params2 = Dict{Symbol, Any}()
	params2[:reco] = "multiCoil"
	params2[:reconSize] = sensSize
	params2[:senseMaps] = sens_spirit;
	
	params2[:solver] = "fista"
	params2[:sparseTrafoName] = "Wavelet"
	params2[:regularization] = "L1"
	params2[:λ] = 0.1 # 5.e-2
	params2[:iterations] = 30
	params2[:normalize_ρ] = true
	params2[:ρ] = 0.07
	#params2[:relTol] = 0.1
	params2[:normalizeReg] = true
	
	
	@time I_wav = reconstruction(acq, params2);
	MP2_wav = mp2rage(I_wav);
	#heatmap(MP2_wav[:,:,slice],c=:grays,aspect_ratio = 1,legend = :none , axis=nothing)
end

# ╔═╡ 18902e8d-2e7f-44ab-90ec-f3d7a5b71ddc
md" Best parameters for now
```julia
	params2[:solver] = \"fista\"
	params2[:sparseTrafoName] = \"Wavelet\"
	params2[:regularization] = \"L1\"
	params2[:λ] = 0.1 # 5.e-2
	params2[:iterations] = 30
	params2[:normalize_ρ] = true
	params2[:ρ] = 0.07
	#params2[:relTol] = 0.1
	params2[:normalizeReg] = true
```
"

# ╔═╡ cbf7f8eb-b4de-42d8-bf38-ef767d5e1157
begin
	plot_vec2 = Any[]
	push!(plot_vec2,heatmap( (MP2_sense[:,:,slice,1,1]), clims = (-0.5, 0.5), c=:grays, aspect_ratio = 1,legend = :none , axis=nothing));
	push!(plot_vec2,heatmap( (MP2_wav[:,:,slice,1,1]), clims = (-0.5, 0.5), c=:grays, aspect_ratio = 1,legend = :none , axis=nothing));
	plot(plot_vec2...)
end

# ╔═╡ 060cc6cf-7a15-4a20-9635-ff7aaed5f8d9
md"# Comparison to BART"

# ╔═╡ 7eda17f1-63ba-42c9-a888-264fe221c927
begin
	k_bart = extract3DKSpace(acq);
	sx,sy,sz,numEcho,numChan = size(k_bart)
	k_bart = permutedims(k_bart,(1,2,3,5,4));
	k_bart = reshape(k_bart,sx,sy,sz,numChan,1,numEcho );
	
	im_pics = bart.pics(kspace = k_bart,sensitivities = sens_spirit,R="W:7:0:0.01",i = 30, S = 1,e = 1);
	im_pics = permutedims(im_pics,(1,2,3,6,4,5));
	im_pics = im_pics[:,:,:,:,:,1];
	
	MP2_pics = mp2rage(im_pics);
end

# ╔═╡ 2755be10-cdd2-4f45-8799-0ef668bf1199
begin
	plot_vec = Any[]
	for echo = 1:2
	  push!(plot_vec,heatmap( abs.(Isense[:,:,slice,echo,1]), c=:grays, aspect_ratio = 1,legend = :none , axis=nothing));
	  push!(plot_vec,heatmap( abs.(I_wav[:,:,slice,echo,1]), c=:grays, aspect_ratio = 1,legend = :none , axis=nothing));
	  push!(plot_vec,heatmap( abs.(im_pics[:,:,slice,echo,1]), c=:grays, aspect_ratio = 1,legend = :none , axis=nothing));
	end
	push!(plot_vec,heatmap( MP2_sense[:,:,slice,1,1],clims = (-0.5, 0.5), c=:grays, aspect_ratio = 1,legend = :none , axis=nothing));
	push!(plot_vec,heatmap( MP2_wav[:,:,slice,1,1], clims = (-0.5, 0.5),c=:grays, aspect_ratio = 1,legend = :none , axis=nothing));
	push!(plot_vec,heatmap( MP2_pics[:,:,slice,1,1],clims = (-0.5, 0.5), c=:grays, aspect_ratio = 1,legend = :none , axis=nothing));
	
plot!(size=(1000,1000))
p_2 = plot(plot_vec...,layout = (3,3))

p_2.subplots[1].attr[:title] = "MRIReco standard"
p_2.subplots[2].attr[:title] = "MRIReco FISTA"
p_2.subplots[3].attr[:title] = "BART"
p_2
end

# ╔═╡ 7b2d823d-e0ba-4ba6-ab8b-d7a994953fd8
md"Results of Fista are not enough regularized. Weirdly I can't do better"

# ╔═╡ 12224ef2-c79a-4f26-8512-3751af09a2ce
md"# Try ADMM ?"

# ╔═╡ 1bcad3dc-b4ae-488a-95cd-24511dff078a
begin
	###############################
	## Reconstruction Parameters ##
	###############################
	params3 = Dict{Symbol, Any}()
	params3[:reco] = "multiCoil"
	params3[:reconSize] = Tuple(acq.encodingSize[1:3])
	params3[:solver] = "admm"
	params3[:regularization] = "L1"
	params3[:sparseTrafo] = "Wavelet"
	params3[:λ] = 1.e-1
	params3[:iterations] = 30
	params3[:ρ] = 1
	params3[:absTol] = 1.e-2
	params3[:relTol] = 1.e-2
	params3[:tolInner] = 1.e-2
	params3[:senseMaps] = sens_spirit
	params3[:normalizeReg] = true

	@time I_admm = reconstruction(acq, params3);
	MP2_admm = mp2rage(I_admm);
end

# ╔═╡ ddfca8d9-f3c8-4887-84c9-533082b98dad
begin
plot_vec3 = Any[]
for echo = 1:2
	push!(plot_vec3,heatmap( abs.(I_wav[:,:,slice,echo,1]), c=:grays, aspect_ratio = 1,legend = :none , axis=nothing));
	push!(plot_vec3,heatmap( abs.(im_pics[:,:,slice,echo,1]), c=:grays, aspect_ratio = 1,legend = :none , axis=nothing));
	push!(plot_vec3,heatmap( abs.(I_admm[:,:,slice,echo,1]), c=:grays, aspect_ratio = 1,legend = :none , axis=nothing));
end
push!(plot_vec3,heatmap(MP2_wav[:,:,slice,1,1], clims = (-0.5, 0.5),c=:grays, aspect_ratio = 1,legend = :none , axis=nothing));
push!(plot_vec3,heatmap(MP2_pics[:,:,slice,1,1],clims = (-0.5, 0.5), c=:grays, aspect_ratio = 1,legend = :none , axis=nothing));
push!(plot_vec3,heatmap(MP2_admm[:,:,slice,1,1],clims = (-0.5, 0.5), c=:grays, aspect_ratio = 1,legend = :none , axis=nothing));

plot!(size=(1000,1000))
p_3 = plot(plot_vec3...,layout = (3,3))

p_3.subplots[1].attr[:title] = "MRIReco FISTA"
p_3.subplots[2].attr[:title] = "BART"
p_3.subplots[3].attr[:title] = "MRIReco ADMM"
p_3
end

# ╔═╡ 6fdff2ae-c8c9-436c-88d8-fa6786027c82
md"# Temporary conclusion

## FISTA
- I have some trouble to find the right parameters for FISTA. I am not able to reach the same image quality as (MRIReci : ADMM / BART)
- Low stability of the results according to some modification of the FISTA parameters 

## Reconstruction time
- Reconstruction time takes longer on my laptop with Julia (11s for bart with all the extract command and 70s for Julia)
- ADMM is particularly long > 500 sec (maybe an issue with my memory/laptop)
- Maybe MRIReco is slower due to F64 vs F32 for bart ?
	"

# ╔═╡ 703593a6-ed0d-4b52-b485-497348298a7f
md"# FISTA over-regularization"

# ╔═╡ ab88b29b-3fe9-41f0-96e5-dc4a1d1c0c5f
begin
	λ = [2.0 1.0 0.5 0.1]
	#λ = [2.0 0.5]
	
	I_wav_λ = Vector{Any}(undef,length(λ))
	MP2_wav_λ = Vector{Any}(undef,length(λ))
	
	for i = 1 :length(λ)
		params2 = Dict{Symbol, Any}()
		params2[:reco] = "multiCoil"
		params2[:reconSize] = sensSize
		params2[:senseMaps] = sens_spirit;
		
		params2[:solver] = "fista"
		params2[:sparseTrafoName] = "Wavelet"
		params2[:regularization] = "L1"
		params2[:λ] = λ[i] # 5.e-2
		params2[:iterations] = 30
		params2[:normalize_ρ] = true
		params2[:ρ] = 0.07
		#params2[:relTol] = 0.1
		params2[:normalizeReg] = true
		
		
		I_wav_λ[i] = reconstruction(acq, params2);
		MP2_wav_λ[i] = mp2rage(I_wav_λ[i]);
	end
end

# ╔═╡ d17698ab-a402-46f6-a14c-3760d4f76918
begin
plot_vec_λ = Any[]
p = Any[]
for i =1 : length(λ)
	for echo = 1:2
		push!(plot_vec_λ,heatmap( abs.(I_wav_λ[i][:,:,slice,echo,1]), c=:grays, aspect_ratio = 1,legend = :none , axis=nothing));
	end
	push!(plot_vec_λ,heatmap(MP2_wav_λ[i][:,:,slice,1,1], clims = (-0.5, 0.5),c=:grays, aspect_ratio = 1,legend = :none , axis=nothing));
	
end

	p = plot(plot_vec_λ...,layouts=(length(λ),3))
	plot!(size=(500,500))
for i =1 : length(λ)
	tmp = λ[i]
	plot!(subplot=(i-1)*3+1, ylabel="λ = $tmp")
end
	p
end

# ╔═╡ e07f8e38-b881-4df2-ae53-bc45c928e026
md"Overregularization generates 0 in images rather than reducing the noise level"

# ╔═╡ Cell order:
# ╟─5fd0776a-3deb-4c70-8bd3-3323876e4902
# ╟─af69fdc6-7abc-45f6-b5a9-59e5fd8deacb
# ╠═ca84f880-8370-11ec-2ed9-1fd70af33535
# ╟─2903d8af-3b4b-4ad4-b480-037a37817b62
# ╠═30e21633-d8d4-49f1-9fa6-4c922a43d222
# ╠═0c47b8c5-76a4-4128-a525-b7b6f7b2c1fc
# ╟─3fb00049-3b01-49dc-9f60-66a989a2508e
# ╠═92eef8c2-55d2-4db2-9967-9a381ae5440b
# ╟─8a459674-94d3-4dd3-b3ea-7b4be4acee98
# ╠═bb732ec5-6eff-4baf-8b6e-204cb21bf61d
# ╟─1e5213aa-c4e1-422e-8600-0c00978a975c
# ╠═d951482f-d45c-4471-9d21-1fbae3bfb678
# ╠═554f16f7-3592-43ab-b90a-d60b90075384
# ╟─f20cb182-2c8b-4892-ae00-54601571a06c
# ╠═50323e44-338c-48f0-a2f3-b024571638c2
# ╟─18902e8d-2e7f-44ab-90ec-f3d7a5b71ddc
# ╠═cbf7f8eb-b4de-42d8-bf38-ef767d5e1157
# ╟─060cc6cf-7a15-4a20-9635-ff7aaed5f8d9
# ╠═c75e0cb0-6220-4dcd-b8d6-26a6f1a64f28
# ╠═7eda17f1-63ba-42c9-a888-264fe221c927
# ╠═2755be10-cdd2-4f45-8799-0ef668bf1199
# ╟─7b2d823d-e0ba-4ba6-ab8b-d7a994953fd8
# ╟─12224ef2-c79a-4f26-8512-3751af09a2ce
# ╠═1bcad3dc-b4ae-488a-95cd-24511dff078a
# ╟─ddfca8d9-f3c8-4887-84c9-533082b98dad
# ╠═6fdff2ae-c8c9-436c-88d8-fa6786027c82
# ╠═703593a6-ed0d-4b52-b485-497348298a7f
# ╠═ab88b29b-3fe9-41f0-96e5-dc4a1d1c0c5f
# ╟─d17698ab-a402-46f6-a14c-3760d4f76918
# ╟─e07f8e38-b881-4df2-ae53-bc45c928e026
