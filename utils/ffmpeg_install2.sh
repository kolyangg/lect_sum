# FFMPEG installation script


# Source: https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu

# 1) Get dependencies
sudo apt-get update -qq && sudo apt-get -y install \
  autoconf \
  automake \
  build-essential \
  cmake \
  git-core \
  libass-dev \
  libfreetype6-dev \
  libgnutls28-dev \
  libmp3lame-dev \
  libsdl2-dev \
  libtool \
  libva-dev \
  libvdpau-dev \
  libvorbis-dev \
  libxcb1-dev \
  libxcb-shm0-dev \
  libxcb-xfixes0-dev \
  meson \
  ninja-build \
  pkg-config \
  texinfo \
  wget \
  yasm \
  zlib1g-dev

sudo apt install libunistring-dev libaom-dev libdav1d-dev

mkdir -p ~/ffmpeg_sources ~/bin

# 2) Install


# d) libvpx
cd ~/ffmpeg_sources && \
git -C libvpx pull 2> /dev/null || git clone --depth 1 https://chromium.googlesource.com/webm/libvpx.git && \
cd libvpx && \
PATH="$HOME/bin:$PATH" ./configure --prefix="$HOME/ffmpeg_build" --disable-examples --disable-unit-tests --enable-vp9-highbitdepth --as=yasm && \
PATH="$HOME/bin:$PATH" make && \
make install


# f) libfdk-aac
cd ~/ffmpeg_sources && \
git -C fdk-aac pull 2> /dev/null || git clone --depth 1 https://github.com/mstorsjo/fdk-aac && \
cd fdk-aac && \
autoreconf -fiv && \
./configure --prefix="$HOME/ffmpeg_build" --disable-shared && \
make && \
make install


# 3) Final install


cd ~/ffmpeg_sources && \
wget -O ffmpeg-snapshot.tar.bz2 https://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2 && \
tar xjvf ffmpeg-snapshot.tar.bz2 && \
cd ffmpeg && \
PATH="$HOME/bin:$PATH" PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" ./configure \
  --prefix="$HOME/ffmpeg_build" \
  --pkg-config-flags="--static" \
  --extra-cflags="-I$HOME/ffmpeg_build/include" \
  --extra-ldflags="-L$HOME/ffmpeg_build/lib" \
  --extra-libs="-lpthread -lm" \
  --ld="g++" \
  --bindir="$HOME/bin" \
  --enable-libfdk-aac \
  --enable-libvorbis \
  --enable-libvpx \
  --enable-nonfree && \
PATH="$HOME/bin:$PATH" make && \
make install && \
hash -r


# cd ~/ffmpeg_sources && \
# wget -O ffmpeg-snapshot.tar.bz2 https://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2 && \
# tar xjvf ffmpeg-snapshot.tar.bz2 && \
# cd ffmpeg && \
# PATH="$HOME/bin:$PATH" PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" ./configure \
#   --prefix="$HOME/ffmpeg_build" \
#   --pkg-config-flags="--static" \
#   --extra-cflags="-I$HOME/ffmpeg_build/include" \
#   --extra-ldflags="-L$HOME/ffmpeg_build/lib" \
#   --extra-libs="-lpthread -lm" \
#   --ld="g++" \
#   --bindir="$HOME/bin" \
#   --enable-gpl \
#   --enable-gnutls \
#   --enable-libass \
#   --enable-libfdk-aac \
#   --enable-libfreetype \
#   --enable-libmp3lame \
#   --enable-libvorbis \
#   --enable-libvpx \
#   --enable-libx265 \
#   --enable-nonfree && \
# PATH="$HOME/bin:$PATH" make && \
# make install && \
# hash -r


# 4) Reogin to recognize new ffmpeg location
source ~/.profile