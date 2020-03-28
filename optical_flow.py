import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageSequence,Image
from skimage.util import view_as_windows,view_as_blocks


# returns a (512,512,n_frames) shaped array
def load_gif_to_array(gif_path="seq.gif"):
    im = Image.open(gif_path)
    im.thumbnail((512,512))
    video = np.zeros([im.size[0], im.size[1], im.n_frames])
    for i, frame in enumerate(ImageSequence.Iterator(im)):
        video[:, :, i] = np.array(frame, dtype="float") / 255
    return video


#returns a (patch_idx_x,patch_idx_y,pixel_idx_x,pixel_idx_y,n_frames) shaped array
def slice_vid_to_patches(video_array,N=16):
    frames_count = video_array.shape[2]
    shape = [video_array.shape[0] // N,video_array.shape[1] // N,N,N,frames_count]
    patches = np.zeros(shape)
    for i in range(frames_count):
        patches[:,:,:,:,i] = view_as_blocks(video_array[:,:,i],(N,N))
    return patches

def derive_vid(vid_array):
    x_back_padded_vid_array = np.pad(vid_array,((0,0),(1,0),(0,0)),"constant")
    x_front_padded_vid_array = np.pad(vid_array,((0,0),(0,1),(0,0)),"constant")
    y_back_padded_vid_array = np.pad(vid_array,((1,0),(0,0),(0,0)),"constant")
    y_front_padded_vid_array = np.pad(vid_array,((0,1),(0,0),(0,0)),"constant")
    t_back_padded_vid_array = np.pad(vid_array,((0,0),(0,0),(1,0)),"constant")
    t_front_padded_vid_array = np.pad(vid_array,((0,0),(0,0),(0,1)),"constant")
    I_x = x_front_padded_vid_array - x_back_padded_vid_array
    I_y = y_front_padded_vid_array - y_back_padded_vid_array
    I_t = t_front_padded_vid_array - t_back_padded_vid_array
    I_x = I_x[:,1:,:]
    I_y = I_y[1:,:,:]
    I_t = I_t[:,:,:-1]
    return I_x,I_y,I_t

def build_and_solve_movement(I_x,I_y,I_t,N,threshold=1):
    I_x_patches = slice_vid_to_patches(I_x,N)
    I_y_patches = slice_vid_to_patches(I_y,N)
    I_t_patches = slice_vid_to_patches(I_t,N)
    patches_count = I_x_patches.shape[0]
    frame_count = I_x_patches.shape[4]
    optical_flow_vid = np.zeros([patches_count,patches_count,2,frame_count])
    for frame in range(frame_count):
        for i in range(patches_count):
            for j in range(patches_count):
                I_x_col = I_x_patches[i,j,:,:,frame].flatten()
                I_y_col = I_y_patches[i,j,:,:,frame].flatten()
                I_t_col = I_t_patches[i,j,:,:,frame].reshape([-1,1])
                A = np.array([I_x_col,I_y_col]).T
                b = -1*I_t_col
                try:
                    harris = np.matmul(A.T,A)
                    eigen_vals = np.linalg.eigvals(harris)
                    if np.min(eigen_vals) > threshold:
                        u = np.matmul(np.matmul(np.linalg.inv(harris),A.T),b).flatten()
                        optical_flow_vid[i,j,:,frame] = u
                    else:
                        optical_flow_vid[i,j,:,frame] = np.zeros([2])
                except:
                    optical_flow_vid[i,j,:,frame] = np.zeros([2])
    return optical_flow_vid


def update_q(frame,frame_count,optical_flow,video,ax,Q):
    f = (frame + 1) % frame_count
    xdim, ydim = video.shape[0], video.shape[1]
    idx_x = np.arange(xdim)
    idx_y = np.arange(ydim)
    #idx_x, idx_y = np.meshgrid(idx_x, idx_y)
    u = optical_flow[:, :, 0, f]
    v = optical_flow[:, :, 1, f]
    #mask = np.logical_or(u != 0, v != 0)  ## <-- corrected: one operation less
    ax.pcolormesh(video[:,:,f], cmap="gray")
    Q.set_UVC(u,v)

    print("frame animated: ", f)
    return Q


def animate(i, video, im, Q, optical_flow):
    a_video = video[:,:,i]
    im.set_data(a_video)
    u = optical_flow[:, :, 0, i]
    v = optical_flow[:, :, 1, i]
    Q.set_UVC(u,v)
    return (im,)

def video_quiver(video,optical_flow,N,threshold):
    frame_count = optical_flow.shape[3]
    xdim,ydim = video.shape[0],video.shape[1]

    idx_x = np.arange(xdim)
    idx_y = np.arange(ydim)
    idx_x, idx_y = np.meshgrid(idx_x, idx_y)
    frame = 0
    u = optical_flow[:, :, 0, frame]
    v = optical_flow[:, :, 1, frame]

    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(video[:,:,0], cmap="gray")
    Q = ax.quiver(idx_x[::N,::N],idx_y[::N,::N],u,v, pivot='tail', color="r",units="xy",scale=0.1)
    from matplotlib import animation
    anim = animation.FuncAnimation(fig, animate, frames=frame_count, fargs=(video,im,Q,optical_flow),blit=False, interval=40)
    fig.tight_layout()
    plt.title("optical flow of video N=" + str(N) + " threshold= " + str(threshold))
    anim.save("seq_of.gif")
    plt.show()
    #return fig, ax, anim


def kl_optical_flow_calc(video_path,N,thresold):
    vid = load_gif_to_array(video_path)
    I_x,I_y,I_t = derive_vid(vid)
    optical_flow = build_and_solve_movement(I_x,I_y,I_t,N,thresold)
    return optical_flow


if __name__ == "__main__":
    N = 16
    threshold = 0.5
    video = load_gif_to_array("seq.gif")
    of = kl_optical_flow_calc("seq.gif",N,threshold)
    video_quiver(video,of,N,threshold)