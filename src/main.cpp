#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <GL/gl.h>
#include <GL/glext.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <numbers>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/geometry.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <thread>
#include <vector>

static PFNGLCREATESHADERPROC glCreateShader = nullptr;
static PFNGLSHADERSOURCEPROC glShaderSource = nullptr;
static PFNGLCOMPILESHADERPROC glCompileShader = nullptr;
static PFNGLGETSHADERIVPROC glGetShaderiv = nullptr;
static PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLog = nullptr;
static PFNGLCREATEPROGRAMPROC glCreateProgram = nullptr;
static PFNGLATTACHSHADERPROC glAttachShader = nullptr;
static PFNGLLINKPROGRAMPROC glLinkProgram = nullptr;
static PFNGLGETPROGRAMIVPROC glGetProgramiv = nullptr;
static PFNGLGETPROGRAMINFOLOGPROC glGetProgramInfoLog = nullptr;
static PFNGLDELETESHADERPROC glDeleteShader = nullptr;
static PFNGLDELETEPROGRAMPROC glDeleteProgram = nullptr;
static PFNGLUSEPROGRAMPROC glUseProgram = nullptr;
static PFNGLGENVERTEXARRAYSPROC glGenVertexArrays = nullptr;
static PFNGLBINDVERTEXARRAYPROC glBindVertexArray = nullptr;
static PFNGLDELETEVERTEXARRAYSPROC glDeleteVertexArrays = nullptr;
static PFNGLGENBUFFERSPROC glGenBuffers = nullptr;
static PFNGLBINDBUFFERPROC glBindBuffer = nullptr;
static PFNGLBUFFERDATAPROC glBufferData = nullptr;
static PFNGLBUFFERSUBDATAPROC glBufferSubData = nullptr;
static PFNGLDELETEBUFFERSPROC glDeleteBuffers = nullptr;
static PFNGLENABLEVERTEXATTRIBARRAYPROC glEnableVertexAttribArray = nullptr;
static PFNGLVERTEXATTRIBPOINTERPROC glVertexAttribPointer = nullptr;
static PFNGLGETUNIFORMLOCATIONPROC glGetUniformLocation = nullptr;
static PFNGLUNIFORM1IPROC glUniform1i = nullptr;
static PFNGLUNIFORM1FPROC glUniform1f = nullptr;
static PFNGLUNIFORM2FPROC glUniform2f = nullptr;
static PFNGLACTIVETEXTUREPROC glActiveTexture = nullptr;

bool load_gl_functions() {
  auto load = [](const char* name) {
    return reinterpret_cast<void*>(glfwGetProcAddress(name));
  };
  glCreateShader =
      reinterpret_cast<PFNGLCREATESHADERPROC>(load("glCreateShader"));
  glShaderSource =
      reinterpret_cast<PFNGLSHADERSOURCEPROC>(load("glShaderSource"));
  glCompileShader =
      reinterpret_cast<PFNGLCOMPILESHADERPROC>(load("glCompileShader"));
  glGetShaderiv = reinterpret_cast<PFNGLGETSHADERIVPROC>(load("glGetShaderiv"));
  glGetShaderInfoLog =
      reinterpret_cast<PFNGLGETSHADERINFOLOGPROC>(load("glGetShaderInfoLog"));
  glCreateProgram =
      reinterpret_cast<PFNGLCREATEPROGRAMPROC>(load("glCreateProgram"));
  glAttachShader =
      reinterpret_cast<PFNGLATTACHSHADERPROC>(load("glAttachShader"));
  glLinkProgram = reinterpret_cast<PFNGLLINKPROGRAMPROC>(load("glLinkProgram"));
  glGetProgramiv =
      reinterpret_cast<PFNGLGETPROGRAMIVPROC>(load("glGetProgramiv"));
  glGetProgramInfoLog =
      reinterpret_cast<PFNGLGETPROGRAMINFOLOGPROC>(load("glGetProgramInfoLog"));
  glDeleteShader =
      reinterpret_cast<PFNGLDELETESHADERPROC>(load("glDeleteShader"));
  glDeleteProgram =
      reinterpret_cast<PFNGLDELETEPROGRAMPROC>(load("glDeleteProgram"));
  glUseProgram = reinterpret_cast<PFNGLUSEPROGRAMPROC>(load("glUseProgram"));
  glGenVertexArrays =
      reinterpret_cast<PFNGLGENVERTEXARRAYSPROC>(load("glGenVertexArrays"));
  glBindVertexArray =
      reinterpret_cast<PFNGLBINDVERTEXARRAYPROC>(load("glBindVertexArray"));
  glDeleteVertexArrays = reinterpret_cast<PFNGLDELETEVERTEXARRAYSPROC>(
      load("glDeleteVertexArrays"));
  glGenBuffers = reinterpret_cast<PFNGLGENBUFFERSPROC>(load("glGenBuffers"));
  glBindBuffer = reinterpret_cast<PFNGLBINDBUFFERPROC>(load("glBindBuffer"));
  glBufferData = reinterpret_cast<PFNGLBUFFERDATAPROC>(load("glBufferData"));
  glBufferSubData =
      reinterpret_cast<PFNGLBUFFERSUBDATAPROC>(load("glBufferSubData"));
  glDeleteBuffers =
      reinterpret_cast<PFNGLDELETEBUFFERSPROC>(load("glDeleteBuffers"));
  glEnableVertexAttribArray =
      reinterpret_cast<PFNGLENABLEVERTEXATTRIBARRAYPROC>(
          load("glEnableVertexAttribArray"));
  glVertexAttribPointer = reinterpret_cast<PFNGLVERTEXATTRIBPOINTERPROC>(
      load("glVertexAttribPointer"));
  glGetUniformLocation = reinterpret_cast<PFNGLGETUNIFORMLOCATIONPROC>(
      load("glGetUniformLocation"));
  glUniform1i = reinterpret_cast<PFNGLUNIFORM1IPROC>(load("glUniform1i"));
  glUniform1f = reinterpret_cast<PFNGLUNIFORM1FPROC>(load("glUniform1f"));
  glUniform2f = reinterpret_cast<PFNGLUNIFORM2FPROC>(load("glUniform2f"));
  glActiveTexture =
      reinterpret_cast<PFNGLACTIVETEXTUREPROC>(load("glActiveTexture"));
  return glCreateShader && glCreateProgram && glGenVertexArrays &&
         glGenBuffers && glActiveTexture && glUniform2f;
}

template <typename T>
class LockFreeTripleBuffer {
 private:
  alignas(64) T slots[3];
  std::atomic<uint8_t> shared_state{1};
  uint8_t write_idx = 0;
  uint8_t read_idx = 2;

 public:
  T& get_write_slot() { return slots[write_idx]; }

  void publish_write() {
    uint8_t next_state = (write_idx & 0x03) | 0x04;
    uint8_t prev_state =
        shared_state.exchange(next_state, std::memory_order_acq_rel);
    write_idx = prev_state & 0x03;
  }

  bool acquire_read_slot(const T*& out_slot) {
    uint8_t current = shared_state.load(std::memory_order_acquire);
    if ((current & 0x04) == 0) {
      out_slot = &slots[read_idx];
      return false;
    }
    while ((current & 0x04) != 0) {
      uint8_t desired = read_idx & 0x03;
      if (shared_state.compare_exchange_weak(current, desired,
                                             std::memory_order_acq_rel,
                                             std::memory_order_acquire)) {
        read_idx = current & 0x03;
        out_slot = &slots[read_idx];
        return true;
      }
    }
    out_slot = &slots[read_idx];
    return false;
  }
};

struct FrameSlot {
  std::vector<uint8_t> pixels;
  int width = 0;
  int height = 0;
  uint64_t sequence_id = 0;
  double timestamp = 0.0;

  void ensure_capacity(int w, int h) {
    size_t required = static_cast<size_t>(w) * static_cast<size_t>(h) * 3;
    if (pixels.size() != required) {
      pixels.resize(required);
    }
    width = w;
    height = h;
  }
};

struct Vec2 {
  float x = 0.0f;
  float y = 0.0f;
};

struct Vec3 {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

struct RawHandData {
  bool detected = false;
  float confidence = 0.0f;
  std::array<Vec3, 21> landmarks{};
};

struct InferencePayload {
  uint64_t sequence_id = 0;
  double timestamp = 0.0;
  std::array<RawHandData, 2> hands{};
};

class OneEuroFilter1D {
 private:
  float min_cutoff;
  float beta;
  float d_cutoff;
  float x_prev = 0.0f;
  float dx_prev = 0.0f;
  bool initialized = false;

  static float smoothing_factor(float dt, float cutoff) {
    float r = 2.0f * std::numbers::pi_v<float> * cutoff * dt;
    return r / (r + 1.0f);
  }

 public:
  OneEuroFilter1D(float min_c = 1.4f, float b = 0.05f, float d_c = 1.0f)
      : min_cutoff(min_c), beta(b), d_cutoff(d_c) {}

  void reset() {
    initialized = false;
    x_prev = 0.0f;
    dx_prev = 0.0f;
  }

  float filter(float x, float dt) {
    if (!initialized || dt <= 0.0001f) {
      initialized = true;
      x_prev = x;
      dx_prev = 0.0f;
      return x;
    }
    float dx = (x - x_prev) / dt;
    float a_d = smoothing_factor(dt, d_cutoff);
    float dx_hat = a_d * dx + (1.0f - a_d) * dx_prev;
    float cutoff = min_cutoff + beta * std::fabs(dx_hat);
    float a = smoothing_factor(dt, cutoff);
    float x_hat = a * x + (1.0f - a) * x_prev;
    x_prev = x_hat;
    dx_prev = dx_hat;
    return x_hat;
  }

  float velocity() const { return dx_prev; }
};

class HandFilterState {
 private:
  std::array<OneEuroFilter1D, 21> fx;
  std::array<OneEuroFilter1D, 21> fy;
  std::array<OneEuroFilter1D, 21> fz;
  double last_measurement_time = 0.0;
  int ttl_frames = 0;
  static constexpr int MAX_TTL = 10;

  float metric_dist(int a, int b) const {
    float dx =
        (smoothed_landmarks[a].x - smoothed_landmarks[b].x) * (16.0f / 9.0f);
    float dy = smoothed_landmarks[a].y - smoothed_landmarks[b].y;
    return std::hypot(dx, dy);
  }

 public:
  std::array<Vec3, 21> smoothed_landmarks{};
  std::array<bool, 5> finger_extended{};
  int extended_count = 0;
  bool active = false;
  bool is_pinched = false;

  static constexpr int TIP_IDS[5] = {4, 8, 12, 16, 20};
  static constexpr int PIP_IDS[5] = {2, 6, 10, 14, 18};
  static constexpr int MCP_IDS[5] = {1, 5, 9, 13, 17};

  void update_from_measurement(const RawHandData& raw,
                               double measurement_time) {
    if (raw.detected) {
      float dt = static_cast<float>(measurement_time - last_measurement_time);
      if (dt <= 0.0f || dt > 0.5f || ttl_frames == 0) {
        dt = 1.0f / 30.0f;
      }
      for (size_t i = 0; i < 21; ++i) {
        smoothed_landmarks[i].x = fx[i].filter(raw.landmarks[i].x, dt);
        smoothed_landmarks[i].y = fy[i].filter(raw.landmarks[i].y, dt);
        smoothed_landmarks[i].z = fz[i].filter(raw.landmarks[i].z, dt);
      }
      last_measurement_time = measurement_time;
      ttl_frames = MAX_TTL;
      active = true;
    } else {
      if (ttl_frames > 0) {
        ttl_frames--;
        active = true;
      } else {
        active = false;
        is_pinched = false;
        extended_count = 0;
        finger_extended.fill(false);
        for (size_t i = 0; i < 21; ++i) {
          fx[i].reset();
          fy[i].reset();
          fz[i].reset();
        }
      }
    }
  }

  void extrapolate_render_tick(float render_dt) {
    if (!active) {
      return;
    }
    float damping = (ttl_frames == MAX_TTL) ? 0.45f : 0.12f;
    float clamped_dt = std::min(render_dt, 0.033f);
    for (size_t i = 0; i < 21; ++i) {
      smoothed_landmarks[i].x = std::clamp(
          smoothed_landmarks[i].x + fx[i].velocity() * clamped_dt * damping,
          0.0f, 1.0f);
      smoothed_landmarks[i].y = std::clamp(
          smoothed_landmarks[i].y + fy[i].velocity() * clamped_dt * damping,
          0.0f, 1.0f);
    }
  }

  void evaluate_fingers() {
    if (!active) {
      extended_count = 0;
      finger_extended.fill(false);
      is_pinched = false;
      return;
    }

    float palm_scale = std::max(0.04f, metric_dist(0, 9));
    float palm_width = std::max(0.03f, metric_dist(5, 17));

    bool thumb_out = (metric_dist(4, 17) > palm_width * 1.22f) &&
                     (metric_dist(4, 9) > palm_scale * 0.50f);
    finger_extended[0] = thumb_out;

    for (int f = 1; f < 5; ++f) {
      int tip = TIP_IDS[f];
      int pip = PIP_IDS[f];
      int mcp = MCP_IDS[f];
      float d_tip = metric_dist(0, tip);
      float d_pip = metric_dist(0, pip);
      float d_mcp = metric_dist(0, mcp);
      float ratio = finger_extended[f] ? 1.06f : 1.14f;
      finger_extended[f] = (d_tip > d_pip * ratio) && (d_tip > d_mcp * 1.25f);
    }

    extended_count = 0;
    for (int f = 0; f < 5; ++f) {
      if (finger_extended[f]) {
        extended_count++;
      }
    }

    float d_pinch = metric_dist(4, 8) / palm_scale;
    if (!is_pinched && d_pinch < 0.34f) {
      is_pinched = true;
    } else if (is_pinched && d_pinch > 0.50f) {
      is_pinched = false;
    }
  }

  Vec2 palm_tilt_offset() const {
    if (!active) {
      return {0.0f, 0.0f};
    }
    float vx = smoothed_landmarks[9].x - smoothed_landmarks[0].x;
    float vy = smoothed_landmarks[9].y - smoothed_landmarks[0].y;
    float wx = smoothed_landmarks[5].x - smoothed_landmarks[17].x;
    float wy = smoothed_landmarks[5].y - smoothed_landmarks[17].y;
    return {vx * 0.22f + wy * 0.18f, vy * 0.22f - wx * 0.18f};
  }
};

class CaptureEngine {
 private:
  LockFreeTripleBuffer<FrameSlot>& render_buffer;
  LockFreeTripleBuffer<FrameSlot>& inference_buffer;
  std::atomic<bool> running{false};
  std::atomic<bool> camera_healthy{false};
  std::thread worker;
  int device_id;

  static double now_sec() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
  }

  void copy_mat_to_slot(const cv::Mat& mat, FrameSlot& slot, uint64_t seq,
                        double ts) {
    slot.ensure_capacity(mat.cols, mat.rows);
    slot.sequence_id = seq;
    slot.timestamp = ts;
    if (mat.isContinuous()) {
      std::memcpy(slot.pixels.data(), mat.data, slot.pixels.size());
    } else {
      size_t row_bytes = static_cast<size_t>(mat.cols) * 3;
      for (int r = 0; r < mat.rows; ++r) {
        std::memcpy(slot.pixels.data() + r * row_bytes, mat.ptr(r), row_bytes);
      }
    }
  }

  void capture_loop() {
    cv::VideoCapture cap;
    uint64_t seq = 0;
    int consecutive_failures = 0;
    cv::Mat raw_frame;
    cv::Mat small_frame;

    while (running.load(std::memory_order_relaxed)) {
      if (!cap.isOpened()) {
        camera_healthy.store(false, std::memory_order_relaxed);
        bool opened = cap.open(device_id, cv::CAP_DSHOW);
        if (!opened) {
          opened = cap.open(device_id, cv::CAP_MSMF);
        }
        if (!opened) {
          std::this_thread::sleep_for(std::chrono::milliseconds(500));
          continue;
        }
        cap.set(cv::CAP_PROP_FOURCC,
                cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        cap.set(cv::CAP_PROP_FPS, 30);
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
      }

      if (!cap.read(raw_frame) || raw_frame.empty() ||
          raw_frame.type() != CV_8UC3) {
        consecutive_failures++;
        if (consecutive_failures > 30) {
          cap.release();
          consecutive_failures = 0;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        continue;
      }

      consecutive_failures = 0;
      camera_healthy.store(true, std::memory_order_relaxed);
      uint64_t current_seq = ++seq;
      double ts = now_sec();

      FrameSlot& r_slot = render_buffer.get_write_slot();
      copy_mat_to_slot(raw_frame, r_slot, current_seq, ts);
      render_buffer.publish_write();

      cv::resize(raw_frame, small_frame, cv::Size(384, 216), 0.0, 0.0,
                 cv::INTER_LINEAR);
      FrameSlot& i_slot = inference_buffer.get_write_slot();
      copy_mat_to_slot(small_frame, i_slot, current_seq, ts);
      inference_buffer.publish_write();
    }
  }

 public:
  CaptureEngine(LockFreeTripleBuffer<FrameSlot>& r_buf,
                LockFreeTripleBuffer<FrameSlot>& i_buf, int dev = 0)
      : render_buffer(r_buf), inference_buffer(i_buf), device_id(dev) {}

  void start() {
    running.store(true, std::memory_order_release);
    worker = std::thread(&CaptureEngine::capture_loop, this);
  }

  void stop() {
    running.store(false, std::memory_order_release);
    if (worker.joinable()) {
      worker.join();
    }
  }

  bool is_healthy() const {
    return camera_healthy.load(std::memory_order_relaxed);
  }

  ~CaptureEngine() { stop(); }
};

struct PalmAnchor {
  float cx, cy;
};

struct DetectedPalm {
  cv::Rect2f bbox;
  float score;
  Vec2 kp_wrist;
  Vec2 kp_middle_mcp;
};

class InferenceEngine {
 private:
  LockFreeTripleBuffer<FrameSlot>& input_buffer;
  LockFreeTripleBuffer<InferencePayload>& output_buffer;
  std::atomic<bool> running{false};
  std::thread worker;
  cv::dnn::Net palm_net;
  cv::dnn::Net landmark_net;
  bool has_two_stage_onnx = false;
  std::vector<PalmAnchor> anchors;
  std::array<DetectedPalm, 2> cached_palms{};
  std::array<bool, 2> has_cached_palm{false, false};

  void generate_ssd_anchors() {
    anchors.clear();
    anchors.reserve(2016);
    for (int y = 0; y < 24; ++y) {
      for (int x = 0; x < 24; ++x) {
        float cx = (static_cast<float>(x) + 0.5f) / 24.0f;
        float cy = (static_cast<float>(y) + 0.5f) / 24.0f;
        anchors.push_back({cx, cy});
        anchors.push_back({cx, cy});
      }
    }
    for (int y = 0; y < 12; ++y) {
      for (int x = 0; x < 12; ++x) {
        float cx = (static_cast<float>(x) + 0.5f) / 12.0f;
        float cy = (static_cast<float>(y) + 0.5f) / 12.0f;
        for (int a = 0; a < 6; ++a) {
          anchors.push_back({cx, cy});
        }
      }
    }
  }

  void init_models() {
    generate_ssd_anchors();
    if (std::filesystem::exists("palm_detection.onnx") &&
        std::filesystem::exists("hand_landmark.onnx")) {
      try {
        palm_net = cv::dnn::readNetFromONNX("palm_detection.onnx");
        palm_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        palm_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        landmark_net = cv::dnn::readNetFromONNX("hand_landmark.onnx");
        landmark_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        landmark_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        has_two_stage_onnx = !palm_net.empty() && !landmark_net.empty();
      } catch (const cv::Exception&) {
        has_two_stage_onnx = false;
      }
    }
  }

  static cv::Mat make_nhwc_blob(const cv::Mat& bgr_img, int target_w,
                                int target_h) {
    cv::Mat resized;
    if (bgr_img.cols == target_w && bgr_img.rows == target_h) {
      resized = bgr_img;
    } else {
      cv::resize(bgr_img, resized, cv::Size(target_w, target_h), 0.0, 0.0,
                 cv::INTER_AREA);
    }
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    cv::Mat float_rgb;
    rgb.convertTo(float_rgb, CV_32FC3, 1.0 / 255.0);
    int dims[4] = {1, target_h, target_w, 3};
    return cv::Mat(4, dims, CV_32F, float_rgb.data).clone();
  }

  bool regress_hand_landmarks(const cv::Mat& bgr_in, const DetectedPalm& p,
                              RawHandData& hand_out,
                              DetectedPalm& updated_palm) {
    float dx = p.kp_middle_mcp.x - p.kp_wrist.x;
    float dy = p.kp_middle_mcp.y - p.kp_wrist.y;
    float cx = p.bbox.x + p.bbox.width * 0.5f + dx * 0.45f;
    float cy = p.bbox.y + p.bbox.height * 0.5f + dy * 0.45f;
    float box_size =
        std::max({p.bbox.width, p.bbox.height, std::hypot(dx, dy) * 2.1f}) *
        2.95f;
    box_size = std::max(box_size, 36.0f);

    double angle_deg = std::atan2(dy, dx) * 180.0 / std::numbers::pi + 90.0;
    double scale = 224.0 / static_cast<double>(box_size);

    cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(cx, cy), angle_deg, scale);
    M.at<double>(0, 2) += 112.0 - cx;
    M.at<double>(1, 2) += 112.0 - cy;

    cv::Mat aligned_crop;
    cv::warpAffine(bgr_in, aligned_crop, M, cv::Size(224, 224),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    cv::Mat lm_blob = make_nhwc_blob(aligned_crop, 224, 224);
    landmark_net.setInput(lm_blob);

    std::vector<std::string> out_names =
        landmark_net.getUnconnectedOutLayersNames();
    std::vector<cv::Mat> lm_outs;
    landmark_net.forward(lm_outs, out_names);

    const cv::Mat* coords_mat = nullptr;
    float presence_conf = 1.0f;
    bool found_first_scalar = false;

    for (size_t k = 0; k < lm_outs.size(); ++k) {
      const auto& m = lm_outs[k];
      if (m.total() == 63) {
        const float* ptr = reinterpret_cast<const float*>(m.data);
        float max_val = 0.0f;
        for (int j = 0; j < 63; ++j) {
          max_val = std::max(max_val, std::fabs(ptr[j]));
        }
        if (max_val > 5.0f) {
          coords_mat = &m;
        }
      } else if (m.total() == 1 && !found_first_scalar) {
        found_first_scalar = true;
        presence_conf = *reinterpret_cast<const float*>(m.data);
      }
    }

    if (!coords_mat || presence_conf < 0.42f) {
      return false;
    }

    cv::Mat M_inv;
    cv::invertAffineTransform(M, M_inv);

    const float* pts = reinterpret_cast<const float*>(coords_mat->data);
    hand_out.detected = true;
    hand_out.confidence = presence_conf;

    float min_px = 1e9f, min_py = 1e9f, max_px = -1e9f, max_py = -1e9f;
    Vec2 img_wrist{}, img_mcp{};

    for (int i = 0; i < 21; ++i) {
      double lx = pts[i * 3 + 0];
      double ly = pts[i * 3 + 1];
      float lz = pts[i * 3 + 2] / 224.0f;

      float ix = static_cast<float>(M_inv.at<double>(0, 0) * lx +
                                    M_inv.at<double>(0, 1) * ly +
                                    M_inv.at<double>(0, 2));
      float iy = static_cast<float>(M_inv.at<double>(1, 0) * lx +
                                    M_inv.at<double>(1, 1) * ly +
                                    M_inv.at<double>(1, 2));

      if (i == 0 || i == 1 || i == 5 || i == 9 || i == 13 || i == 17) {
        min_px = std::min(min_px, ix);
        min_py = std::min(min_py, iy);
        max_px = std::max(max_px, ix);
        max_py = std::max(max_py, iy);
      }
      if (i == 0) img_wrist = {ix, iy};
      if (i == 9) img_mcp = {ix, iy};

      hand_out.landmarks[i] = {
          std::clamp(1.0f - (ix / static_cast<float>(bgr_in.cols)), 0.0f, 1.0f),
          std::clamp(1.0f - (iy / static_cast<float>(bgr_in.rows)), 0.0f, 1.0f),
          lz};
    }

    float pw = std::max(28.0f, (max_px - min_px) * 1.15f);
    float ph = std::max(28.0f, (max_py - min_py) * 1.15f);
    float pcx = 0.5f * (min_px + max_px);
    float pcy = 0.5f * (min_py + max_py);
    updated_palm = {cv::Rect2f(pcx - pw * 0.5f, pcy - ph * 0.5f, pw, ph),
                    presence_conf, img_wrist, img_mcp};
    return true;
  }

  void run_two_stage_onnx(const cv::Mat& bgr_in, InferencePayload& out) {
    int max_side = std::max(bgr_in.cols, bgr_in.rows);
    int pad_w = max_side - bgr_in.cols;
    int pad_h = max_side - bgr_in.rows;
    int left = pad_w / 2;
    int top = pad_h / 2;
    cv::Mat padded;
    cv::copyMakeBorder(bgr_in, padded, top, pad_h - top, left, pad_w - left,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    cv::Mat palm_blob = make_nhwc_blob(padded, 192, 192);
    palm_net.setInput(palm_blob);
    std::vector<cv::Mat> palm_outs;
    palm_net.forward(palm_outs, palm_net.getUnconnectedOutLayersNames());

    if (palm_outs.size() < 2) {
      return;
    }

    const cv::Mat* reg_mat = &palm_outs[0];
    const cv::Mat* cls_mat = &palm_outs[1];
    if (palm_outs[0].total() == 2016) {
      std::swap(reg_mat, cls_mat);
    }

    const float* regs = reinterpret_cast<const float*>(reg_mat->data);
    const float* clss = reinterpret_cast<const float*>(cls_mat->data);

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<DetectedPalm> raw_palms;

    for (size_t i = 0; i < 2016; ++i) {
      float score = 1.0f / (1.0f + std::exp(-clss[i]));
      if (score < 0.44f) {
        continue;
      }
      const float* r = regs + i * 18;
      float cx = anchors[i].cx + r[0] / 192.0f;
      float cy = anchors[i].cy + r[1] / 192.0f;
      float w = r[2] / 192.0f;
      float h = r[3] / 192.0f;

      float x0 = (cx - w * 0.5f) * max_side - left;
      float y0 = (cy - h * 0.5f) * max_side - top;
      float pw = w * max_side;
      float ph = h * max_side;

      float kp0_x = (anchors[i].cx + r[4] / 192.0f) * max_side - left;
      float kp0_y = (anchors[i].cy + r[5] / 192.0f) * max_side - top;
      float kp2_x = (anchors[i].cx + r[8] / 192.0f) * max_side - left;
      float kp2_y = (anchors[i].cy + r[9] / 192.0f) * max_side - top;

      boxes.emplace_back(static_cast<int>(x0), static_cast<int>(y0),
                         static_cast<int>(pw), static_cast<int>(ph));
      scores.push_back(score);
      raw_palms.push_back(
          {cv::Rect2f(x0, y0, pw, ph), score, {kp0_x, kp0_y}, {kp2_x, kp2_y}});
    }

    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(boxes, scores, 0.44f, 0.25f, nms_indices);

    std::vector<DetectedPalm> distinct_palms;
    float min_sep = static_cast<float>(bgr_in.cols) * 0.20f;
    for (int idx : nms_indices) {
      const DetectedPalm& cand = raw_palms[idx];
      float cand_cx = cand.bbox.x + cand.bbox.width * 0.5f;
      float cand_cy = cand.bbox.y + cand.bbox.height * 0.5f;
      bool duplicate = false;
      for (const auto& existing : distinct_palms) {
        float ex_cx = existing.bbox.x + existing.bbox.width * 0.5f;
        float ex_cy = existing.bbox.y + existing.bbox.height * 0.5f;
        if (std::hypot(cand_cx - ex_cx, cand_cy - ex_cy) < min_sep) {
          duplicate = true;
          break;
        }
      }
      if (!duplicate) {
        distinct_palms.push_back(cand);
        if (distinct_palms.size() == 2) {
          break;
        }
      }
    }

    for (size_t s = 0; s < 2; ++s) {
      if (distinct_palms.size() < 2 && has_cached_palm[s]) {
        float ccx = cached_palms[s].bbox.x + cached_palms[s].bbox.width * 0.5f;
        float ccy = cached_palms[s].bbox.y + cached_palms[s].bbox.height * 0.5f;
        bool already_covered = false;
        for (const auto& dp : distinct_palms) {
          float dcx = dp.bbox.x + dp.bbox.width * 0.5f;
          float dcy = dp.bbox.y + dp.bbox.height * 0.5f;
          if (std::hypot(ccx - dcx, ccy - dcy) < min_sep) {
            already_covered = true;
            break;
          }
        }
        if (!already_covered) {
          distinct_palms.push_back(cached_palms[s]);
        }
      }
    }

    has_cached_palm = {false, false};
    for (size_t h = 0; h < distinct_palms.size(); ++h) {
      RawHandData hand{};
      DetectedPalm next_palm{};
      if (regress_hand_landmarks(bgr_in, distinct_palms[h], hand, next_palm)) {
        float palm_screen_x =
            0.5f * (hand.landmarks[0].x + hand.landmarks[9].x);
        size_t slot = (palm_screen_x < 0.5f) ? 0 : 1;
        if (out.hands[slot].detected) {
          slot = 1 - slot;
        }
        if (!out.hands[slot].detected) {
          out.hands[slot] = hand;
          cached_palms[slot] = next_palm;
          has_cached_palm[slot] = true;
        }
      }
    }

    if (out.hands[0].detected && out.hands[1].detected) {
      if (out.hands[0].landmarks[0].x > out.hands[1].landmarks[0].x) {
        std::swap(out.hands[0], out.hands[1]);
        std::swap(cached_palms[0], cached_palms[1]);
      }
    }
  }

  void inference_loop() {
    init_models();
    uint64_t last_processed_seq = 0;

    while (running.load(std::memory_order_relaxed)) {
      const FrameSlot* slot = nullptr;
      bool has_new = input_buffer.acquire_read_slot(slot);
      if (!has_new || !slot || slot->sequence_id == last_processed_seq ||
          slot->width == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        continue;
      }

      last_processed_seq = slot->sequence_id;
      cv::Mat bgr_in(slot->height, slot->width, CV_8UC3,
                     const_cast<uint8_t*>(slot->pixels.data()));

      InferencePayload& out = output_buffer.get_write_slot();
      out.sequence_id = slot->sequence_id;
      out.timestamp = slot->timestamp;
      out.hands[0] = RawHandData{};
      out.hands[1] = RawHandData{};

      if (has_two_stage_onnx) {
        run_two_stage_onnx(bgr_in, out);
      }

      output_buffer.publish_write();
    }
  }

 public:
  InferenceEngine(LockFreeTripleBuffer<FrameSlot>& in_buf,
                  LockFreeTripleBuffer<InferencePayload>& out_buf)
      : input_buffer(in_buf), output_buffer(out_buf) {}

  void start() {
    running.store(true, std::memory_order_release);
    worker = std::thread(&InferenceEngine::inference_loop, this);
  }

  void stop() {
    running.store(false, std::memory_order_release);
    if (worker.joinable()) {
      worker.join();
    }
  }

  ~InferenceEngine() { stop(); }
};

static const char* BG_VS_SRC = R"(
#version 330 core
layout (location = 0) in vec2 a_pos;
layout (location = 1) in vec2 a_uv;
out vec2 v_uv;
void main() {
    v_uv = vec2(1.0 - a_uv.x, 1.0 - a_uv.y);
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
)";

static const char* BG_FS_SRC = R"(
#version 330 core
in vec2 v_uv;
out vec4 frag_color;
uniform sampler2D u_camera_tex;
uniform int u_camera_healthy;
uniform int u_thermal_fullscreen;
uniform float u_time;

vec3 thermal_palette(float t) {
    vec3 c0 = vec3(0.95, 0.0, 0.55);
    vec3 c1 = vec3(0.0, 0.85, 1.0);
    vec3 c2 = vec3(0.1, 1.0, 0.2);
    vec3 c3 = vec3(1.0, 0.95, 0.0);
    if (t < 0.33) return mix(c0, c1, t * 3.0);
    if (t < 0.66) return mix(c1, c2, (t - 0.33) * 3.0);
    return mix(c2, c3, (t - 0.66) * 3.0);
}

void main() {
    if (u_camera_healthy == 0) {
        float pulse = 0.15 + 0.05 * sin(u_time * 4.0);
        frag_color = vec4(pulse, 0.05, 0.05, 1.0);
        return;
    }
    vec3 col = texture(u_camera_tex, v_uv).rgb;
    if (u_thermal_fullscreen == 1) {
        float lum = dot(col, vec3(0.299, 0.587, 0.114));
        frag_color = vec4(thermal_palette(lum), 1.0);
        return;
    }
    frag_color = vec4(col, 1.0);
}
)";

static const char* FX_VS_SRC = R"(
#version 330 core
layout (location = 0) in vec2 a_pos;
layout (location = 1) in vec2 a_local_uv;
layout (location = 2) in float a_face_id;
out vec2 v_screen_uv;
out vec2 v_local_uv;
out float v_face_id;
void main() {
    v_screen_uv = a_pos;
    v_local_uv = a_local_uv;
    v_face_id = a_face_id;
    vec2 ndc = a_pos * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
)";

static const char* FX_FS_SRC = R"(
#version 330 core
in vec2 v_screen_uv;
in vec2 v_local_uv;
in float v_face_id;
out vec4 frag_color;
uniform sampler2D u_camera_tex;
uniform int u_effect_mode;
uniform float u_time;
uniform vec2 u_quad_center;
uniform vec2 u_quad_span;

vec3 thermal_lut(float t) {
    vec3 c0 = vec3(0.35, 0.0, 0.75);
    vec3 c1 = vec3(0.0, 0.45, 1.0);
    vec3 c2 = vec3(0.0, 1.0, 0.45);
    vec3 c3 = vec3(1.0, 0.9, 0.0);
    vec3 c4 = vec3(1.0, 0.1, 0.5);
    if (t < 0.25) return mix(c0, c1, t * 4.0);
    if (t < 0.50) return mix(c1, c2, (t - 0.25) * 4.0);
    if (t < 0.75) return mix(c2, c3, (t - 0.50) * 4.0);
    return mix(c3, c4, (t - 0.75) * 4.0);
}

void main() {
    vec2 cam_uv = vec2(1.0 - v_screen_uv.x, 1.0 - v_screen_uv.y);
    vec3 cam = texture(u_camera_tex, cam_uv).rgb;
    float lum = dot(cam, vec3(0.299, 0.587, 0.114));
    int face = int(v_face_id + 0.5);

    if (face == 1) {
        float stripe = fract((v_screen_uv.x * 160.0 + v_screen_uv.y * 90.0));
        float mask = step(0.45, stripe);
        frag_color = vec4(vec3(mask), 0.88);
        return;
    }

    if (face == 2) {
        vec2 grid = v_screen_uv * vec2(130.0, 74.0);
        float d = length(fract(grid) - 0.5);
        float m = step(0.32, d);
        frag_color = vec4(mix(vec3(0.95, 0.3, 0.9), vec3(0.15, 0.05, 0.55), m), 0.88);
        return;
    }

    if (u_effect_mode == 0) {
        float diag = fract((v_screen_uv.x * 110.0 - v_screen_uv.y * 65.0));
        float hatch = step(lum * 0.85 + 0.1, diag);
        vec3 col = mix(vec3(0.08, 0.15, 0.88), vec3(0.97, 0.98, 1.0), hatch);
        frag_color = vec4(col, 0.95);
    } else if (u_effect_mode == 1) {
        vec2 grid = v_screen_uv * vec2(96.0, 54.0);
        vec2 cell = fract(grid) - 0.5;
        float dist = length(cell);
        float radius = (1.0 - lum) * 0.52;
        float dot_mask = smoothstep(radius + 0.04, radius - 0.04, dist);
        vec3 dot_col = mix(vec3(0.95, 0.05, 0.45), vec3(1.0, 0.65, 0.05), fract(lum * 2.3));
        vec3 col = mix(vec3(0.98, 0.98, 1.0), dot_col, dot_mask);
        frag_color = vec4(col, 0.95);
    } else if (u_effect_mode == 2) {
        vec3 tcol = thermal_lut(lum);
        frag_color = vec4(tcol, 0.94);
    } else if (u_effect_mode == 3) {
        float wave = sin(v_local_uv.x * 8.0 + v_local_uv.y * 6.0 + lum * 7.0 + u_time * 2.5);
        vec3 irid = mix(vec3(0.22, 0.05, 0.75), vec3(0.88, 0.65, 1.0), wave * 0.5 + 0.5);
        irid = mix(irid, vec3(1.0), smoothstep(0.65, 0.95, wave));
        frag_color = vec4(irid, 0.92);
    } else if (u_effect_mode == 4) {
        vec2 center_cam = vec2(1.0 - u_quad_center.x, 1.0 - u_quad_center.y);
        vec2 zoomed_uv = clamp(center_cam + (cam_uv - center_cam) * 0.45, vec2(0.002), vec2(0.998));
        vec2 texel = vec2(1.0 / 1280.0, 1.0 / 720.0);
        vec3 z_col = texture(u_camera_tex, zoomed_uv).rgb;
        float tl = dot(texture(u_camera_tex, zoomed_uv + vec2(-texel.x,  texel.y)).rgb, vec3(0.333));
        float l  = dot(texture(u_camera_tex, zoomed_uv + vec2(-texel.x,  0.0)).rgb,     vec3(0.333));
        float bl = dot(texture(u_camera_tex, zoomed_uv + vec2(-texel.x, -texel.y)).rgb, vec3(0.333));
        float t  = dot(texture(u_camera_tex, zoomed_uv + vec2( 0.0,      texel.y)).rgb, vec3(0.333));
        float b  = dot(texture(u_camera_tex, zoomed_uv + vec2( 0.0,     -texel.y)).rgb, vec3(0.333));
        float tr = dot(texture(u_camera_tex, zoomed_uv + vec2( texel.x,  texel.y)).rgb, vec3(0.333));
        float r  = dot(texture(u_camera_tex, zoomed_uv + vec2( texel.x,  0.0)).rgb,     vec3(0.333));
        float br = dot(texture(u_camera_tex, zoomed_uv + vec2( texel.x, -texel.y)).rgb, vec3(0.333));
        float gx = -tl - 2.0 * l - bl + tr + 2.0 * r + br;
        float gy = -tl - 2.0 * t - tr + bl + 2.0 * b + br;
        float edge = clamp(length(vec2(gx, gy)) * 3.2, 0.0, 1.0);
        vec2 grid = abs(fract(v_local_uv * 8.0) - 0.5);
        float grid_line = smoothstep(0.47, 0.495, max(grid.x, grid.y)) * 0.25;
        vec3 out_c = mix(z_col * 0.85, vec3(0.0, 1.0, 0.85), edge);
        out_c += vec3(0.2, 0.9, 1.0) * grid_line;
        frag_color = vec4(out_c, 0.96);
    } else {
        vec3 bg = mix(vec3(0.03, 0.07, 0.14), cam * 0.35, 0.45);
        vec2 g = abs(fract(v_local_uv * vec2(12.0, 8.0)) - 0.5);
        float g_mask = smoothstep(0.46, 0.495, max(g.x, g.y));
        float freq = clamp(u_quad_span.x * 28.0, 4.0, 24.0);
        float amp = clamp(u_quad_span.y * 0.85, 0.08, 0.42);
        float y_wave1 = 0.5 + amp * sin(v_local_uv.x * freq - u_time * 5.0);
        float y_wave2 = 0.5 + amp * 0.65 * cos(v_local_uv.x * (freq * 1.5) + u_time * 3.5);
        float w1 = smoothstep(0.028, 0.004, abs(v_local_uv.y - y_wave1));
        float w2 = smoothstep(0.022, 0.003, abs(v_local_uv.y - y_wave2));
        float axis = smoothstep(0.012, 0.002, abs(v_local_uv.y - 0.5));
        vec3 col = bg + vec3(0.08, 0.28, 0.45) * g_mask + vec3(0.3, 0.4, 0.5) * axis;
        col = mix(col, vec3(0.1, 1.0, 0.75), w1);
        col = mix(col, vec3(1.0, 0.45, 0.85), w2);
        frag_color = vec4(col, 0.92);
    }
}
)";

static const char* OVERLAY_VS_SRC = R"(
#version 330 core
layout (location = 0) in vec2 a_pos;
layout (location = 1) in vec3 a_color;
out vec3 v_color;
void main() {
    v_color = a_color;
    vec2 ndc = a_pos * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    gl_PointSize = 7.0;
}
)";

static const char* OVERLAY_FS_SRC = R"(
#version 330 core
in vec3 v_color;
out vec4 frag_color;
void main() {
    frag_color = vec4(v_color, 1.0);
}
)";

GLuint compile_shader(GLenum type, const char* src) {
  GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &src, nullptr);
  glCompileShader(shader);
  GLint success = 0;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
  if (!success) {
    char log[512];
    glGetShaderInfoLog(shader, 512, nullptr, log);
    std::cerr << "shader compilation failed: " << log << "\n";
    std::terminate();
  }
  return shader;
}

GLuint create_program(const char* vs_src, const char* fs_src) {
  GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_src);
  GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_src);
  GLuint prog = glCreateProgram();
  glAttachShader(prog, vs);
  glAttachShader(prog, fs);
  glLinkProgram(prog);
  GLint success = 0;
  glGetProgramiv(prog, GL_LINK_STATUS, &success);
  if (!success) {
    char log[512];
    glGetProgramInfoLog(prog, 512, nullptr, log);
    std::cerr << "program linking failed: " << log << "\n";
    std::terminate();
  }
  glDeleteShader(vs);
  glDeleteShader(fs);
  return prog;
}

void framebuffer_size_callback(GLFWwindow*, int width, int height) {
  glViewport(0, 0, width, height);
}

struct OverlayVertex {
  float x, y;
  float r, g, b;
};

struct FxVertex {
  float x, y;
  float u, v;
  float face_id;
};

struct PinnedSimplexState {
  bool active = false;
  std::vector<Vec2> poly;
  Vec2 extrusion_offset{0.04f, -0.06f};
  int effect_mode = 0;
};

void push_line(std::vector<OverlayVertex>& verts, float x0, float y0, float x1,
               float y1, float r = 1.0f, float g = 1.0f, float b = 1.0f) {
  verts.push_back({x0, y0, r, g, b});
  verts.push_back({x1, y1, r, g, b});
}

void push_dashed_line(std::vector<OverlayVertex>& verts, float x0, float y0,
                      float x1, float y1, int segments = 18, float r = 1.0f,
                      float g = 1.0f, float b = 1.0f) {
  for (int i = 0; i < segments; ++i) {
    if (i % 2 != 0) {
      continue;
    }
    float t0 = static_cast<float>(i) / static_cast<float>(segments);
    float t1 = static_cast<float>(i + 1) / static_cast<float>(segments);
    push_line(verts, x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0,
              x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1, r, g, b);
  }
}

void append_reticle(std::vector<OverlayVertex>& verts, float cx, float cy,
                    float half_w, float half_h, float r = 1.0f, float g = 1.0f,
                    float b = 1.0f) {
  float cw = half_w * 0.42f;
  float ch = half_h * 0.42f;
  float x0 = cx - half_w, x1 = cx + half_w;
  float y0 = cy - half_h, y1 = cy + half_h;

  push_line(verts, x0, y0, x0 + cw, y0, r, g, b);
  push_line(verts, x0, y0, x0, y0 + ch, r, g, b);
  push_line(verts, x1, y0, x1 - cw, y0, r, g, b);
  push_line(verts, x1, y0, x1, y0 + ch, r, g, b);
  push_line(verts, x0, y1, x0 + cw, y1, r, g, b);
  push_line(verts, x0, y1, x0, y1 - ch, r, g, b);
  push_line(verts, x1, y1, x1 - cw, y1, r, g, b);
  push_line(verts, x1, y1, x1, y1 - ch, r, g, b);
}

void append_char_glyph(std::vector<OverlayVertex>& verts, char c, float x,
                       float y, float w, float h, float r = 1.0f,
                       float g = 1.0f, float b = 1.0f) {
  float xm = x + w * 0.5f;
  float xr = x + w;
  float ym = y + h * 0.5f;
  float yt = y + h;

  auto seg = [&](int mask) {
    if (mask & 1) push_line(verts, x, yt, xr, yt, r, g, b);
    if (mask & 2) push_line(verts, xr, ym, xr, yt, r, g, b);
    if (mask & 4) push_line(verts, xr, y, xr, ym, r, g, b);
    if (mask & 8) push_line(verts, x, y, xr, y, r, g, b);
    if (mask & 16) push_line(verts, x, y, x, ym, r, g, b);
    if (mask & 32) push_line(verts, x, ym, x, yt, r, g, b);
    if (mask & 64) push_line(verts, x, ym, xr, ym, r, g, b);
    if (mask & 128) push_line(verts, xm, y, xm, yt, r, g, b);
    if (mask & 256) push_line(verts, x, yt, xm, ym, r, g, b);
    if (mask & 512) push_line(verts, xr, yt, xm, ym, r, g, b);
    if (mask & 1024) push_line(verts, x, y, xm, ym, r, g, b);
    if (mask & 2048) push_line(verts, xr, y, xm, ym, r, g, b);
  };

  if (c >= 'a' && c <= 'z') {
    c = static_cast<char>(c - 'a' + 'A');
  }

  switch (c) {
    case '0':
      seg(1 | 2 | 4 | 8 | 16 | 32);
      break;
    case '1':
      seg(128);
      break;
    case '2':
      seg(1 | 2 | 64 | 16 | 8);
      break;
    case '3':
      seg(1 | 2 | 64 | 4 | 8);
      break;
    case '4':
      seg(32 | 64 | 2 | 4);
      break;
    case '5':
      seg(1 | 32 | 64 | 4 | 8);
      break;
    case '6':
      seg(1 | 32 | 16 | 8 | 4 | 64);
      break;
    case '7':
      seg(1 | 2 | 4);
      break;
    case '8':
      seg(1 | 2 | 4 | 8 | 16 | 32 | 64);
      break;
    case '9':
      seg(1 | 2 | 4 | 8 | 32 | 64);
      break;
    case 'A':
      seg(1 | 2 | 4 | 16 | 32 | 64);
      break;
    case 'B':
      seg(1 | 2 | 4 | 8 | 16 | 32 | 64);
      break;
    case 'C':
      seg(1 | 8 | 16 | 32);
      break;
    case 'D':
      seg(1 | 2 | 4 | 8 | 128);
      break;
    case 'E':
      seg(1 | 8 | 16 | 32 | 64);
      break;
    case 'F':
      seg(1 | 16 | 32 | 64);
      break;
    case 'G':
      seg(1 | 4 | 8 | 16 | 32 | 64);
      break;
    case 'H':
      seg(2 | 4 | 16 | 32 | 64);
      break;
    case 'I':
      seg(1 | 8 | 128);
      break;
    case 'L':
      seg(8 | 16 | 32);
      break;
    case 'M':
      seg(2 | 4 | 16 | 32 | 256 | 512);
      break;
    case 'N':
      seg(2 | 4 | 16 | 32 | 256 | 2048);
      break;
    case 'O':
      seg(1 | 2 | 4 | 8 | 16 | 32);
      break;
    case 'P':
      seg(1 | 2 | 16 | 32 | 64);
      break;
    case 'R':
      seg(1 | 2 | 16 | 32 | 64 | 2048);
      break;
    case 'S':
      seg(1 | 32 | 64 | 4 | 8);
      break;
    case 'T':
      seg(1 | 128);
      break;
    case 'U':
      seg(2 | 4 | 8 | 16 | 32);
      break;
    case 'V':
      seg(16 | 32 | 1024 | 512);
      break;
    case 'W':
      seg(2 | 4 | 16 | 32 | 1024 | 2048);
      break;
    case 'X':
      seg(256 | 512 | 1024 | 2048);
      break;
    case 'Y':
      seg(256 | 512 | 128);
      break;
    case '-':
      seg(64);
      break;
    case ':':
      push_line(verts, xm, y + h * 0.2f, xm, y + h * 0.32f, r, g, b);
      push_line(verts, xm, y + h * 0.65f, xm, y + h * 0.77f, r, g, b);
      break;
    default:
      break;
  }
}

void append_text_string(std::vector<OverlayVertex>& verts,
                        const std::string& text, float x, float y, float cw,
                        float ch, float r = 1.0f, float g = 1.0f,
                        float b = 1.0f) {
  float advance = cw * 1.45f;
  float cur_x = x;
  for (char c : text) {
    if (c != ' ') {
      append_char_glyph(verts, c, cur_x, y, cw, ch, r, g, b);
    }
    cur_x += advance;
  }
}

void append_coord_label(std::vector<OverlayVertex>& verts, float x, float y,
                        int px, int py) {
  std::string text = "X:" + std::to_string(px) + " Y:" + std::to_string(py);
  float cur_x = std::clamp(x, 0.02f, 0.80f);
  float cur_y = std::clamp(y, 0.03f, 0.94f);
  append_text_string(verts, text, cur_x, cur_y, 0.0082f, 0.021f);
}

void push_quad_fx(std::vector<FxVertex>& verts, Vec2 p0, Vec2 p1, Vec2 p2,
                  Vec2 p3, float face_id) {
  verts.push_back({p0.x, p0.y, 0.0f, 0.0f, face_id});
  verts.push_back({p1.x, p1.y, 1.0f, 0.0f, face_id});
  verts.push_back({p2.x, p2.y, 1.0f, 1.0f, face_id});
  verts.push_back({p0.x, p0.y, 0.0f, 0.0f, face_id});
  verts.push_back({p2.x, p2.y, 1.0f, 1.0f, face_id});
  verts.push_back({p3.x, p3.y, 0.0f, 1.0f, face_id});
}

void build_ngon_simplex_geometry(const std::vector<Vec2>& poly, Vec2 centroid,
                                 Vec2 offset, int effect_mode,
                                 std::vector<FxVertex>& fx_verts,
                                 std::vector<OverlayVertex>& line_verts,
                                 bool is_pinned) {
  size_t n = poly.size();
  if (n < 3) {
    return;
  }

  float lr = is_pinned ? 0.25f : 1.0f;
  float lg = is_pinned ? 1.0f : 1.0f;
  float lb = is_pinned ? 0.80f : 1.0f;

  float min_x = poly[0].x, max_x = poly[0].x;
  float min_y = poly[0].y, max_y = poly[0].y;
  for (const auto& p : poly) {
    min_x = std::min(min_x, p.x);
    max_x = std::max(max_x, p.x);
    min_y = std::min(min_y, p.y);
    max_y = std::max(max_y, p.y);
  }
  float span_x = std::max(0.001f, max_x - min_x);
  float span_y = std::max(0.001f, max_y - min_y);

  if (effect_mode == 2 || effect_mode == 3) {
    std::vector<Vec2> back_poly(n);
    for (size_t i = 0; i < n; ++i) {
      back_poly[i] = {poly[i].x + offset.x, poly[i].y + offset.y};
    }
    for (size_t i = 0; i < n; ++i) {
      size_t next = (i + 1) % n;
      float side_face =
          (i % 2 == 0) ? 1.0f : ((effect_mode == 3) ? 2.0f : 1.0f);
      push_quad_fx(fx_verts, poly[i], poly[next], back_poly[next], back_poly[i],
                   side_face);
      push_line(line_verts, poly[i].x, poly[i].y, back_poly[i].x,
                back_poly[i].y, lr, lg, lb);
      push_line(line_verts, back_poly[i].x, back_poly[i].y, back_poly[next].x,
                back_poly[next].y, lr, lg, lb);
    }
  }

  float cu = (centroid.x - min_x) / span_x;
  float cv = (centroid.y - min_y) / span_y;
  for (size_t i = 0; i < n; ++i) {
    const Vec2& p1 = poly[i];
    const Vec2& p2 = poly[(i + 1) % n];
    fx_verts.push_back({centroid.x, centroid.y, cu, cv, 0.0f});
    fx_verts.push_back(
        {p1.x, p1.y, (p1.x - min_x) / span_x, (p1.y - min_y) / span_y, 0.0f});
    fx_verts.push_back(
        {p2.x, p2.y, (p2.x - min_x) / span_x, (p2.y - min_y) / span_y, 0.0f});
  }

  for (size_t i = 0; i < n; ++i) {
    size_t next = (i + 1) % n;
    push_line(line_verts, poly[i].x, poly[i].y, poly[next].x, poly[next].y, lr,
              lg, lb);
  }
  if (n >= 4) {
    push_dashed_line(line_verts, poly[0].x, poly[0].y, poly[n / 2].x,
                     poly[n / 2].y, 16, lr, lg, lb);
    push_dashed_line(line_verts, poly[1].x, poly[1].y, poly[(1 + n / 2) % n].x,
                     poly[(1 + n / 2) % n].y, 16, lr, lg, lb);
  }
}

int main() {
  if (!glfwInit()) {
    std::cerr << "failed to initialize glfw\n";
    return -1;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);

  GLFWwindow* window = glfwCreateWindow(1152, 648, "simplex", nullptr, nullptr);
  if (!window) {
    std::cerr << "failed to create glfw window\n";
    glfwTerminate();
    return -1;
  }

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

  if (!load_gl_functions()) {
    std::cerr << "failed to load opengl 3.3 functions\n";
    glfwTerminate();
    return -1;
  }

  int fb_w = 0, fb_h = 0;
  glfwGetFramebufferSize(window, &fb_w, &fb_h);
  glViewport(0, 0, fb_w, fb_h);

  glEnable(GL_PROGRAM_POINT_SIZE);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glLineWidth(1.8f);

  GLuint bg_program = create_program(BG_VS_SRC, BG_FS_SRC);
  GLuint fx_program = create_program(FX_VS_SRC, FX_FS_SRC);
  GLuint overlay_program = create_program(OVERLAY_VS_SRC, OVERLAY_FS_SRC);

  float quad_vertices[] = {-1.0f, 1.0f,  0.0f, 1.0f, -1.0f, -1.0f, 0.0f, 0.0f,
                           1.0f,  -1.0f, 1.0f, 0.0f, -1.0f, 1.0f,  0.0f, 1.0f,
                           1.0f,  -1.0f, 1.0f, 0.0f, 1.0f,  1.0f,  1.0f, 1.0f};

  GLuint bg_vao = 0, bg_vbo = 0;
  glGenVertexArrays(1, &bg_vao);
  glGenBuffers(1, &bg_vbo);
  glBindVertexArray(bg_vao);
  glBindBuffer(GL_ARRAY_BUFFER, bg_vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(quad_vertices), quad_vertices,
               GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                        (void*)(2 * sizeof(float)));

  GLuint fx_vao = 0, fx_vbo = 0;
  glGenVertexArrays(1, &fx_vao);
  glGenBuffers(1, &fx_vbo);
  glBindVertexArray(fx_vao);
  glBindBuffer(GL_ARRAY_BUFFER, fx_vbo);
  glBufferData(GL_ARRAY_BUFFER, 4096 * sizeof(FxVertex), nullptr,
               GL_DYNAMIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(FxVertex), (void*)0);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(FxVertex),
                        (void*)(2 * sizeof(float)));
  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(FxVertex),
                        (void*)(4 * sizeof(float)));

  GLuint overlay_vao = 0, overlay_vbo = 0;
  glGenVertexArrays(1, &overlay_vao);
  glGenBuffers(1, &overlay_vbo);
  glBindVertexArray(overlay_vao);
  glBindBuffer(GL_ARRAY_BUFFER, overlay_vbo);
  glBufferData(GL_ARRAY_BUFFER, 32768 * sizeof(OverlayVertex), nullptr,
               GL_DYNAMIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(OverlayVertex),
                        (void*)0);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(OverlayVertex),
                        (void*)(2 * sizeof(float)));

  GLuint camera_tex = 0;
  glGenTextures(1, &camera_tex);
  glBindTexture(GL_TEXTURE_2D, camera_tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  int allocated_tex_w = 0;
  int allocated_tex_h = 0;

  LockFreeTripleBuffer<FrameSlot> render_frame_buf;
  LockFreeTripleBuffer<FrameSlot> inference_frame_buf;
  LockFreeTripleBuffer<InferencePayload> landmark_buf;

  CaptureEngine capture(render_frame_buf, inference_frame_buf, 0);
  InferenceEngine inference(inference_frame_buf, landmark_buf);

  capture.start();
  inference.start();

  GLint loc_bg_tex = glGetUniformLocation(bg_program, "u_camera_tex");
  GLint loc_bg_healthy = glGetUniformLocation(bg_program, "u_camera_healthy");
  GLint loc_bg_thermal =
      glGetUniformLocation(bg_program, "u_thermal_fullscreen");
  GLint loc_bg_time = glGetUniformLocation(bg_program, "u_time");

  GLint loc_fx_tex = glGetUniformLocation(fx_program, "u_camera_tex");
  GLint loc_fx_mode = glGetUniformLocation(fx_program, "u_effect_mode");
  GLint loc_fx_time = glGetUniformLocation(fx_program, "u_time");
  GLint loc_fx_center = glGetUniformLocation(fx_program, "u_quad_center");
  GLint loc_fx_span = glGetUniformLocation(fx_program, "u_quad_span");

  std::array<HandFilterState, 2> hand_filters{};
  std::vector<std::vector<Vec2>> strokes;
  bool is_currently_drawing = false;
  PinnedSimplexState pinned_simplex{};

  int current_effect_mode = 0;
  static constexpr int TOTAL_MODES = 6;
  static const char* MODE_NAMES[TOTAL_MODES] = {
      "M1:HATCH",     "M2:HALFTONE",   "M3:3D-THERM",
      "M4:3D-CHROME", "M5:LOUPE-EDGE", "M6:WAVE-GRID"};

  bool auto_cycle_modes = false;
  bool prev_a_pressed = false;
  bool prev_p_pressed = false;
  double last_mode_switch_time = 0.0;
  double last_clear_time = 0.0;

  std::vector<Vec2> last_live_poly;
  Vec2 last_live_offset{0.04f, -0.06f};

  double prev_render_time = glfwGetTime();
  std::vector<FxVertex> fx_verts;
  std::vector<OverlayVertex> line_verts;
  fx_verts.reserve(1024);
  line_verts.reserve(16384);

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
      glfwSetWindowShouldClose(window, true);
    }

    double current_time = glfwGetTime();
    float render_dt = static_cast<float>(current_time - prev_render_time);
    prev_render_time = current_time;

    if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) {
      current_effect_mode = 0;
      auto_cycle_modes = false;
    }
    if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) {
      current_effect_mode = 1;
      auto_cycle_modes = false;
    }
    if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) {
      current_effect_mode = 2;
      auto_cycle_modes = false;
    }
    if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS) {
      current_effect_mode = 3;
      auto_cycle_modes = false;
    }
    if (glfwGetKey(window, GLFW_KEY_5) == GLFW_PRESS) {
      current_effect_mode = 4;
      auto_cycle_modes = false;
    }
    if (glfwGetKey(window, GLFW_KEY_6) == GLFW_PRESS) {
      current_effect_mode = 5;
      auto_cycle_modes = false;
    }

    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) {
      strokes.clear();
      pinned_simplex.active = false;
    }

    bool a_pressed = (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS);
    if (a_pressed && !prev_a_pressed) {
      auto_cycle_modes = !auto_cycle_modes;
      last_mode_switch_time = current_time;
    }
    prev_a_pressed = a_pressed;

    bool p_pressed = (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS);
    if (p_pressed && !prev_p_pressed) {
      if (pinned_simplex.active) {
        pinned_simplex.active = false;
      } else if (last_live_poly.size() >= 3) {
        pinned_simplex.active = true;
        pinned_simplex.poly = last_live_poly;
        pinned_simplex.extrusion_offset = last_live_offset;
        pinned_simplex.effect_mode = current_effect_mode;
      }
    }
    prev_p_pressed = p_pressed;

    if (auto_cycle_modes && (current_time - last_mode_switch_time > 3.5)) {
      current_effect_mode = (current_effect_mode + 1) % TOTAL_MODES;
      last_mode_switch_time = current_time;
    }

    const FrameSlot* read_slot = nullptr;
    bool has_new_frame = render_frame_buf.acquire_read_slot(read_slot);

    if (has_new_frame && read_slot && read_slot->width > 0 &&
        read_slot->height > 0) {
      glBindTexture(GL_TEXTURE_2D, camera_tex);
      if (read_slot->width != allocated_tex_w ||
          read_slot->height != allocated_tex_h) {
        allocated_tex_w = read_slot->width;
        allocated_tex_h = read_slot->height;
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, allocated_tex_w,
                     allocated_tex_h, 0, GL_BGR, GL_UNSIGNED_BYTE,
                     read_slot->pixels.data());
      } else {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, allocated_tex_w,
                        allocated_tex_h, GL_BGR, GL_UNSIGNED_BYTE,
                        read_slot->pixels.data());
      }
    }

    const InferencePayload* inf_slot = nullptr;
    if (landmark_buf.acquire_read_slot(inf_slot) && inf_slot) {
      for (size_t h = 0; h < 2; ++h) {
        hand_filters[h].update_from_measurement(inf_slot->hands[h],
                                                inf_slot->timestamp);
      }
    } else {
      for (size_t h = 0; h < 2; ++h) {
        hand_filters[h].extrapolate_render_tick(render_dt);
      }
    }

    bool both_active = hand_filters[0].active && hand_filters[1].active;
    std::vector<Vec2> active_vertices;
    active_vertices.reserve(10);

    for (size_t h = 0; h < 2; ++h) {
      hand_filters[h].evaluate_fingers();
      if (!hand_filters[h].active) {
        continue;
      }
      for (int f = 0; f < 5; ++f) {
        if (hand_filters[h].finger_extended[f]) {
          int tip_id = HandFilterState::TIP_IDS[f];
          Vec2 pt = {hand_filters[h].smoothed_landmarks[tip_id].x,
                     hand_filters[h].smoothed_landmarks[tip_id].y};
          active_vertices.push_back(pt);

          if (pt.x > 0.80f && pt.y > 0.84f &&
              (current_time - last_mode_switch_time > 0.70)) {
            current_effect_mode = (current_effect_mode + 1) % TOTAL_MODES;
            last_mode_switch_time = current_time;
          }
          if (pt.x > 0.80f && pt.y > 0.72f && pt.y <= 0.83f &&
              (current_time - last_clear_time > 0.70)) {
            strokes.clear();
            pinned_simplex.active = false;
            last_clear_time = current_time;
          }
        }
      }
    }

    bool single_index_pen = false;
    Vec2 pen_pt{};
    for (size_t h = 0; h < 2; ++h) {
      if (hand_filters[h].active && !both_active) {
        bool index_only = hand_filters[h].finger_extended[1] &&
                          !hand_filters[h].finger_extended[2] &&
                          !hand_filters[h].finger_extended[3] &&
                          !hand_filters[h].finger_extended[4] &&
                          (!hand_filters[h].finger_extended[0] ||
                           (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS));
        if (index_only) {
          single_index_pen = true;
          pen_pt = {hand_filters[h].smoothed_landmarks[8].x,
                    hand_filters[h].smoothed_landmarks[8].y};
        }
      }
    }

    if (single_index_pen) {
      if (!is_currently_drawing || strokes.empty()) {
        strokes.push_back({pen_pt});
        is_currently_drawing = true;
      } else {
        auto& cur_stroke = strokes.back();
        if (std::hypot(pen_pt.x - cur_stroke.back().x,
                       pen_pt.y - cur_stroke.back().y) > 0.008f) {
          if (cur_stroke.size() < 512) {
            cur_stroke.push_back(pen_pt);
          }
        }
      }
    } else {
      is_currently_drawing = false;
    }

    bool thermal_overload =
        (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) ||
        (both_active && hand_filters[0].is_pinched &&
         hand_filters[1].is_pinched);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(bg_program);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, camera_tex);
    glUniform1i(loc_bg_tex, 0);
    glUniform1i(loc_bg_healthy,
                (capture.is_healthy() && allocated_tex_w > 0) ? 1 : 0);
    glUniform1i(loc_bg_thermal, thermal_overload ? 1 : 0);
    glUniform1f(loc_bg_time, static_cast<float>(current_time));
    glBindVertexArray(bg_vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    fx_verts.clear();
    line_verts.clear();

    append_reticle(line_verts, 0.5f, 0.5f, 0.46f, 0.44f);

    std::string status_str =
        std::string(MODE_NAMES[current_effect_mode]) +
        " - V:" + std::to_string(active_vertices.size()) +
        (pinned_simplex.active ? " - PINNED"
                               : (auto_cycle_modes ? " - AUTO" : " - LIVE"));
    append_text_string(line_verts, status_str, 0.06f, 0.90f, 0.0085f, 0.023f,
                       0.35f, 1.0f, 0.85f);

    append_reticle(line_verts, 0.86f, 0.88f, 0.055f, 0.035f, 0.35f, 1.0f,
                   0.85f);
    append_text_string(line_verts, "MODE", 0.825f, 0.87f, 0.0075f, 0.019f,
                       0.35f, 1.0f, 0.85f);

    append_reticle(line_verts, 0.86f, 0.77f, 0.055f, 0.035f, 1.0f, 0.45f,
                   0.65f);
    append_text_string(line_verts, "CLR", 0.835f, 0.76f, 0.0075f, 0.019f, 1.0f,
                       0.45f, 0.65f);

    for (const auto& stroke : strokes) {
      for (size_t i = 1; i < stroke.size(); ++i) {
        push_line(line_verts, stroke[i - 1].x, stroke[i - 1].y, stroke[i].x,
                  stroke[i].y, 0.15f, 1.0f, 0.95f);
      }
    }

    Vec2 active_quad_center{0.5f, 0.5f};
    Vec2 active_quad_span{0.35f, 0.25f};
    int shader_mode_to_draw = current_effect_mode;

    if (active_vertices.size() == 2 && !thermal_overload) {
      push_line(line_verts, active_vertices[0].x, active_vertices[0].y,
                active_vertices[1].x, active_vertices[1].y, 1.0f, 1.0f, 1.0f);
    } else if (active_vertices.size() >= 3 && !thermal_overload) {
      Vec2 centroid{0.0f, 0.0f};
      for (const auto& v : active_vertices) {
        centroid.x += v.x;
        centroid.y += v.y;
      }
      centroid.x /= static_cast<float>(active_vertices.size());
      centroid.y /= static_cast<float>(active_vertices.size());

      std::sort(active_vertices.begin(), active_vertices.end(),
                [&](const Vec2& a, const Vec2& b) {
                  return std::atan2(a.y - centroid.y, a.x - centroid.x) <
                         std::atan2(b.y - centroid.y, b.x - centroid.x);
                });

      Vec2 t0 = hand_filters[0].palm_tilt_offset();
      Vec2 t1 = hand_filters[1].palm_tilt_offset();
      Vec2 dyn_offset = {
          std::clamp((centroid.x - 0.5f) * 0.22f + (t0.x + t1.x) * 0.5f, -0.14f,
                     0.14f),
          std::clamp((centroid.y - 0.5f) * 0.22f + (t0.y + t1.y) * 0.5f - 0.04f,
                     -0.14f, 0.14f)};

      last_live_poly = active_vertices;
      last_live_offset = dyn_offset;

      float min_x = active_vertices[0].x, max_x = active_vertices[0].x;
      float min_y = active_vertices[0].y, max_y = active_vertices[0].y;
      for (const auto& v : active_vertices) {
        min_x = std::min(min_x, v.x);
        max_x = std::max(max_x, v.x);
        min_y = std::min(min_y, v.y);
        max_y = std::max(max_y, v.y);
      }

      active_quad_center = centroid;
      active_quad_span = {max_x - min_x, max_y - min_y};
      shader_mode_to_draw = current_effect_mode;

      build_ngon_simplex_geometry(active_vertices, centroid, dyn_offset,
                                  current_effect_mode, fx_verts, line_verts,
                                  false);
    } else if (pinned_simplex.active && pinned_simplex.poly.size() >= 3 &&
               !thermal_overload) {
      const auto& poly = pinned_simplex.poly;
      Vec2 centroid{0.0f, 0.0f};
      float min_x = poly[0].x, max_x = poly[0].x;
      float min_y = poly[0].y, max_y = poly[0].y;
      for (const auto& v : poly) {
        centroid.x += v.x;
        centroid.y += v.y;
        min_x = std::min(min_x, v.x);
        max_x = std::max(max_x, v.x);
        min_y = std::min(min_y, v.y);
        max_y = std::max(max_y, v.y);
      }
      centroid.x /= static_cast<float>(poly.size());
      centroid.y /= static_cast<float>(poly.size());

      active_quad_center = centroid;
      active_quad_span = {max_x - min_x, max_y - min_y};
      shader_mode_to_draw = pinned_simplex.effect_mode;

      build_ngon_simplex_geometry(
          poly, centroid, pinned_simplex.extrusion_offset,
          pinned_simplex.effect_mode, fx_verts, line_verts, true);
      for (const auto& v : poly) {
        append_reticle(line_verts, v.x, v.y, 0.022f, 0.036f, 0.25f, 1.0f,
                       0.80f);
      }
    }

    for (size_t h = 0; h < 2; ++h) {
      if (!hand_filters[h].active) {
        continue;
      }
      const auto& lms = hand_filters[h].smoothed_landmarks;
      for (int f = 0; f < 5; ++f) {
        if (!hand_filters[h].finger_extended[f]) {
          continue;
        }
        int tip_id = HandFilterState::TIP_IDS[f];
        Vec2 pt = {lms[tip_id].x, lms[tip_id].y};
        append_reticle(line_verts, pt.x, pt.y, 0.024f, 0.038f);
        if (f == 0 || f == 1) {
          append_coord_label(line_verts, pt.x + 0.028f, pt.y - 0.015f,
                             static_cast<int>(pt.x * 960.0f),
                             static_cast<int>((1.0f - pt.y) * 540.0f));
        }
      }
    }

    if (!fx_verts.empty()) {
      glUseProgram(fx_program);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, camera_tex);
      glUniform1i(loc_fx_tex, 0);
      glUniform1i(loc_fx_mode, shader_mode_to_draw);
      glUniform1f(loc_fx_time, static_cast<float>(current_time));
      glUniform2f(loc_fx_center, active_quad_center.x, active_quad_center.y);
      glUniform2f(loc_fx_span, active_quad_span.x, active_quad_span.y);
      glBindVertexArray(fx_vao);
      glBindBuffer(GL_ARRAY_BUFFER, fx_vbo);
      size_t count = std::min<size_t>(fx_verts.size(), 4096);
      glBufferSubData(GL_ARRAY_BUFFER, 0, count * sizeof(FxVertex),
                      fx_verts.data());
      glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(count));
    }

    if (!line_verts.empty()) {
      glUseProgram(overlay_program);
      glBindVertexArray(overlay_vao);
      glBindBuffer(GL_ARRAY_BUFFER, overlay_vbo);
      size_t count = std::min<size_t>(line_verts.size(), 32768);
      glBufferSubData(GL_ARRAY_BUFFER, 0, count * sizeof(OverlayVertex),
                      line_verts.data());
      glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(count));
    }

    glfwSwapBuffers(window);
  }

  inference.stop();
  capture.stop();
  glDeleteTextures(1, &camera_tex);
  glDeleteBuffers(1, &overlay_vbo);
  glDeleteVertexArrays(1, &overlay_vao);
  glDeleteBuffers(1, &fx_vbo);
  glDeleteVertexArrays(1, &fx_vao);
  glDeleteBuffers(1, &bg_vbo);
  glDeleteVertexArrays(1, &bg_vao);
  glDeleteProgram(overlay_program);
  glDeleteProgram(fx_program);
  glDeleteProgram(bg_program);
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}