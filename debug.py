
import numpy as np
import matplotlib.pyplot as plt

def maxKseparation_debug(obj):
	Kᵧ = obj.Kᵧ
	Q = obj.Q
	ɡ = obj.ɡ
	ḡ = obj.ḡ
	Ⅱᵀ = obj.Ⅱᵀ
	ƌₐ = []
	ƌᵦ = []
	lossⲷ = []
	σⲷ = np.arange(0.01,7, 0.01)

	for σ in σⲷ:
		Kₓ = np.exp(obj.Ðᒾ/(σ*σ))
		ƌₐ.append(ɡ*np.sum(Kₓ*Kᵧ))
		ƌᵦ.append(ḡ*np.sum(Kₓ*(Ⅱᵀ - Kᵧ)))

		Δƌ = np.array(ƌₐ) - np.array(ƌᵦ)
		lossⲷ.append(obj.maxKseparation(σ))

	loss = obj.maxKseparation(obj.result.x)
	lossₒ = obj.maxKseparation(obj.σₒ)

	print('σₒ = %.3f'%obj.σₒ)
	print('σ = %.3f'%obj.result.x)
	print('lossₒ = %.3f'%lossₒ)
	print('loss = %.3f'%loss)

	optσText = 'Optimal σ : %.3f\nopt kernel separation : %.3f'%(obj.result.x, -loss)

	plt.plot(σⲷ, ƌₐ, 'r-')
	plt.plot(σⲷ, ƌᵦ, 'b-')
	plt.plot(σⲷ, Δƌ, 'g-')
	#plt.plot(σⲷ, lossⲷ, 'y-')
	plt.xlabel('σ value')
	plt.ylabel('Kernel Value')
	plt.title('Kernel Value as Varying σ')
	plt.text(σⲷ[-1], ƌₐ[-1], 'Mean within cluster kernel value', horizontalalignment='right')
	plt.text(σⲷ[-1], ƌᵦ[-1], 'Mean between cluster kernel value', horizontalalignment='right')
	plt.text(obj.result.x, -loss, optσText, horizontalalignment='center')
	plt.axvline(x=obj.result.x, linestyle="dashed")

	#plt.text(σⲷ[-1], Δƌ[-1], 'Kernel Value Δ', horizontalalignment='right')

	plt.show()


def ℍ_debug(obj):
	Kᵧ = obj.Kᵧ
	Q = obj.Q
	ɡ = obj.ɡ
	ḡ = obj.ḡ
	Ⲏ = obj.Ⲏ
	Ⅱᵀ = obj.Ⅱᵀ
	ƌₐ = []
	ƌᵦ = []
	lossⲷ = []
	σⲷ = np.arange(0.01,7, 0.01)

	for σ in σⲷ:

		Kₓ = np.exp(obj.Ðᒾ/(σ*σ))
		Ƙ = (Kₓ.dot(Ⲏ))*(Kᵧ.dot(Ⲏ))

		ƌₐ.append(np.sum(Ƙ*Kᵧ))
		ƌᵦ.append(np.sum(Ƙ*(Ⅱᵀ - Kᵧ)))

		lossⲷ.append(-obj.ℍ(σ))

	Δƌ = np.array(ƌₐ) - np.array(ƌᵦ)

	loss = obj.ℍ(obj.result.x)
	lossₒ = obj.ℍ(obj.σₒ)

	print('σₒ = %.3f'%obj.σₒ)
	print('σ = %.3f'%obj.result.x)
	print('lossₒ = %.3f'%lossₒ)
	print('loss = %.3f'%loss)

	optσText = 'Optimal σ : %.3f\nopt kernel separation : %.3f'%(obj.result.x, -loss)

	plt.plot(σⲷ, ƌₐ, 'r-')
	plt.plot(σⲷ, ƌᵦ, 'b-')
	#plt.plot(σⲷ, Δƌ, 'g-')
	plt.plot(σⲷ, lossⲷ, 'y-')
	plt.xlabel('σ value')
	plt.ylabel('HSIC Value')
	plt.title('HSIC as Varying σ')
	plt.text(σⲷ[-1], ƌₐ[-1], 'Mean within cluster kernel value', horizontalalignment='right')
	plt.text(σⲷ[-1], ƌᵦ[-1], 'Mean between cluster kernel value', horizontalalignment='right')
	plt.text(obj.result.x, -loss, optσText, horizontalalignment='center', va='top')
	plt.axvline(x=obj.result.x, linestyle="dashed")

	#plt.text(σⲷ[-1], Δƌ[-1], 'Kernel Value Δ', horizontalalignment='right')

	plt.show()

