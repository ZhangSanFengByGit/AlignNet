

import torch
from DownLayer import DownLayer
from kron import KronEmbed
from UpLayer import UpLayer

class AlignNet(torch.nn.Module):
	def __init__(self):
		super(AlignNet,self).__init__()
		self.down1x = DownLayer(3,16)
		self.down2x = DownLayer(16,64)
		self.down3x = DownLayer(64,128)
		self.down4x = DownLayer(128,256)
        
		self.down1y = DownLayer(3,16)
		self.down2y = DownLayer(16,64)
		self.down3y = DownLayer(64,128)
		self.down4y = DownLayer(128,256)

		#kron 部分的源码还有待修改，其现在的版本包含了不止matching，masked的部分
		self.kron = KronEmbed()

		self.up1 = UpLayer(256,128)
		self.up2 = UpLayer(128,64)
		self.up3 = UpLayer(64,16)
		self.up4 = UpLayer(16,3)

	def forward(self,img_x,img_y):
		out_x = self.down1x(img_x)
		out_y = self.down1y(img_y)
		out_x = self.down2x(out_x)
		out_y = self.down2y(out_y)
		out_x = self.down3x(out_x)
		out_y = self.down3y(out_y)
		out_x = self.down4x(out_x)
		out_y = self.down4y(out_y)

		#这里注意清楚是masked谁，后期联合kron一并修改
		masked_y = self.kron(out_x,out_y)

		out = self.up1(masked_y)
		out = self.up2(out)
		out = self.up3(out)
		out = self.up4(out)

		return out


		






