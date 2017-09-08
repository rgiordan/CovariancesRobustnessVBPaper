# Sizes for images with multiple plots in them.  These sizes can be used
# inside a knitr chunk definition.  NB: I did this before I realized that
# grid.arrange now works with knitr.  That is a better way than this.

# These are based on one image per row.
aspect_ratio <- 3.5 / (5 * 2)
image_width <- 4.9 * 2

# A list for standardizing the size of images.
imsize <- list()

im1 <- list()
im1$ow <- "0.98\\linewidth"
im1$oh <- sprintf("%0.3f\\linewidth", aspect_ratio * 0.98)
im1$fw <- image_width
im1$fh <- image_width * aspect_ratio

im2h <- list()
im2h$ow <- "0.98\\linewidth"
im2h$oh <- sprintf("%0.3f\\linewidth", 2 * aspect_ratio * 0.98)
im2h$fw <- image_width
im2h$fh <- 2 * image_width * aspect_ratio


# Make the default a one image..
opts_chunk$set(out.width=im1$ow, out.height=im1$oh, fig.width=im1$fw, fig.height=im1$fh)
