import gradio as gr
import os
import json
import hashlib
import requests
import time
import shutil
import tempfile
import cv2
import base64
from shared.utils.plugins import WAN2GPPlugin

CIVITAI_HOST = "https://civitai.com"
TRPC_URL = "https://civitai.com/api/trpc/model.getAll"
REST_URL = "https://civitai.com/api/v1/models"
IMAGE_BASE_URL = "https://imagecache.civitai.com/xG1nkqKTMzGDvpLrqFT7WA"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

SORT_OPTIONS = ["Highest Rated", "Most Downloaded", "Newest", "Most Liked", "Most Discussed", "Most Collected"]
PERIOD_OPTIONS = ["AllTime", "Year", "Month", "Week", "Day"]
MODEL_TYPES = [
    "Checkpoint", "LORA", "TextualInversion", "Hypernetwork", 
    "AestheticGradient", "Controlnet", "Poses", "Wildcards", 
    "Workflows", "MotionModule", "VAE", "Upscaler", "LoCon", "DoRA", "Detection", "Other"
]
BASE_MODELS = [
    "AuraFlow", "Chroma", "CogVideoX", "Flux.1 S", "Flux.1 D", "Flux.1 Krea",
    "Flux.1 Kontext", "Flux.2 D", "HiDream", "Hunyuan 1", "Hunyuan Video",
    "Illustrious", "Kolors", "LTXV", "Lumina", "Mochi", "NoobAI", "Other",
    "PixArt a", "PixArt E", "Pony", "Pony V7", "Qwen", "SD 1.4", "SD 1.5",
    "SD 1.5 LCM", "SD 1.5 Hyper", "SD 2.0", "SD 2.1", "SDXL 1.0",
    "SDXL Lightning", "SDXL Hyper", "Wan Video 1.3B t2v", "Wan Video 14B t2v",
    "Wan Video 14B i2v 480p", "Wan Video 14B i2v 720p", "Wan Video 2.2 TI2V-5B",
    "Wan Video 2.2 I2V-A14B", "Wan Video 2.2 T2V-A14B", "Wan Video 2.5 T2V",
    "Wan Video 2.5 I2V", "ZImageTurbo"
]
DEFAULT_BASE_SELECTION = []

CIVIT_TO_WANGP_ARCH = {
    "Wan Video 14B t2v": "t2v",
    "Wan Video 1.3B t2v": "t2v_1.3B",
    "Wan Video 14B i2v 480p": "i2v",
    "Wan Video 14B i2v 720p": "i2v",
    "Wan Video 2.2 T2V-A14B": "t2v",
    "Wan Video 2.2 I2V-A14B": "t2v",
    "Wan Video 2.2 TI2V-5B": "ti2v_2_2",
    "Hunyuan Video": "hunyuan_1_5_t2v",
    "Hunyuan 1": "hunyuan",
    "Flux.1 D": "flux",
    "Flux.1 S": "flux_schnell",
    "Flux.1 Krea": "flux",
    "Flux.1 Kontext": "flux_dev_kontext",
    "Flux.2 D": "flux2_dev",
    "LTXV": "ltxv_13B",
    "Qwen": "qwen_image_20B",
    "ZImageTurbo": "z_image",
    "Mochi": "mocha"
}

PLACEHOLDER_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 24 24" fill="none" stroke="#4b5563" stroke-width="1" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>"""
PLACEHOLDER_B64 = f"data:image/svg+xml;base64,{base64.b64encode(PLACEHOLDER_SVG.encode('utf-8')).decode('utf-8')}"

class LoraManagerPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "LoRA Manager"
        self.version = "2.5.5"
        self.description = "Manage local LoRAs with lset support and granular prompt/file injection."
        
        self.plugin_dir = os.path.dirname(os.path.abspath(__file__))
        self.lora_root = "loras" 
        self.finetunes_root = "finetunes"
        self.metadata_root = "loras_metadata"
        
        self.settings_path = os.path.join(self.plugin_dir, "settings.json")
        
        self.previews_dir = os.path.abspath(os.path.join("icons", "lora_previews"))
        os.makedirs(self.previews_dir, exist_ok=True)
        os.makedirs(self.metadata_root, exist_ok=True)
        
        if not os.path.exists(self.finetunes_root):
            try: os.makedirs(self.finetunes_root, exist_ok=True)
            except: pass

        self.saved_settings = {}
        self.items_cache = []
        self.cursor_cache = None
        self.manager_to_browser_btn = None
        self.manager_to_browser_state = None
        self.civit_tabs = None
        self.api_key = None

    def setup_ui(self):
        self.request_global("get_lora_dir")
        self.request_global("get_state_model_type")
        self.request_global("model_types") 
        self.request_global("get_model_name") 

        self.request_component("state")
        self.request_component("prompt") 
        self.request_component("loras_choices")
        self.request_component("main_tabs")
        
        self.load_settings()
        
        self.on_tab_outputs = [] 

        self.add_custom_js("""
            window.civitSelectCard = function(id) {
                const container = document.getElementById('civit_bridge_input');
                if (!container) return;
                const textarea = container.querySelector('textarea');
                if (!textarea) return;
                textarea.value = id;
                textarea.dispatchEvent(new Event('input', { bubbles: true }));
            }

            window.triggerManagerToBrowser = function() {
                const btn = document.getElementById('manager_to_browser_btn');
                if (btn) btn.click();
            }

            window.toggleLoraCard = function(el, filename) {
                el.classList.toggle('selected');

                const bridge = document.getElementById('lora_selection_bridge');
                if (!bridge) return console.error('Bridge not found');
                const textarea = bridge.querySelector('textarea');
                
                let current = [];
                try { current = JSON.parse(textarea.value || "[]"); } catch(e) {}
                
                if (current.includes(filename)) {
                    current = current.filter(x => x !== filename);
                } else {
                    current.push(filename);
                }
                
                textarea.value = JSON.stringify(current);
                textarea.dispatchEvent(new Event('input', { bubbles: true }));
            }

            let civitScrollTimeout;
            window.addEventListener('scroll', () => {
                clearTimeout(civitScrollTimeout);
                civitScrollTimeout = setTimeout(() => {
                    const h = document.documentElement;
                    if ((h.scrollTop + h.clientHeight) > (h.scrollHeight - 500)) {
                        const btn = document.getElementById('civit_load_more_btn');
                        if(btn) btn.click();
                    }
                }, 150);
            });
        """)

        self.add_tab(
            tab_id="lora_manager_tab",
            label="LoRA Manager",
            component_constructor=self.create_manager_ui,
            position=2
        )

        self.add_tab(
            tab_id="civitai_browser_tab",
            label="CivitAI Browser",
            component_constructor=self.create_browser_ui,
            position=3
        )

    def load_settings(self):
        if os.path.exists(self.settings_path):
            try:
                with open(self.settings_path, 'r', encoding='utf-8') as f:
                    self.saved_settings = json.load(f)
            except:
                self.saved_settings = {}
        else:
            self.saved_settings = {}

    def save_settings_to_disk(self, **kwargs):
        for k, v in kwargs.items():
            self.saved_settings[k] = v
        try:
            with open(self.settings_path, 'w', encoding='utf-8') as f:
                json.dump(self.saved_settings, f, indent=4)
        except Exception as e: print(f"Error saving settings: {e}")

    def get_headers(self, api_key: str = "") -> dict:
        headers = {
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json",
            "Referer": "https://civitai.com/models"
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def construct_media_url(self, uuid: str, width: int = 450, is_video: bool = False) -> str:
        filename = "preview.mp4" if is_video else "preview.jpg"
        return f"{IMAGE_BASE_URL}/{uuid}/width={width}/{filename}"

    def get_local_preview_path(self, model_id):
        if not model_id: return None, None
        for ext in ['.jpg', '.png', '.mp4', '.webm']:
            filename = f"{model_id}{ext}"
            local_path = os.path.join(self.previews_dir, filename)
            if os.path.exists(local_path):
                return local_path, f"/gradio_api/file={local_path}"
        return None, None

    def download_preview_image(self, url, model_id):
        if not url: return
        mode = self.saved_settings.get("preview_mode", "Image Thumbnail (First Frame)")
        
        try:
            is_video = url.endswith('.mp4') or url.endswith('.webm')
            ext = os.path.splitext(url)[1] if is_video else '.jpg'

            if is_video and mode == "Original Media (Video/Image)":
                target = os.path.join(self.previews_dir, f"{model_id}{ext}")
            else:
                target = os.path.join(self.previews_dir, f"{model_id}.jpg")

            r = requests.get(url, headers=self.get_headers(), stream=True)
            if r.status_code != 200: return

            if is_video:
                if mode == "Original Media (Video/Image)":
                    with open(target, 'wb') as f:
                        shutil.copyfileobj(r.raw, f)
                else:
                    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_vid:
                        shutil.copyfileobj(r.raw, tmp_vid)
                        tmp_vid_path = tmp_vid.name
                    
                    try:
                        cap = cv2.VideoCapture(tmp_vid_path)
                        ret, frame = cap.read()
                        if ret: cv2.imwrite(target, frame)
                        cap.release()
                    except Exception as e: print(f"Frame extract error: {e}")
                    finally:
                        if os.path.exists(tmp_vid_path): os.remove(tmp_vid_path)
            else:
                with open(target, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
        except Exception as e: print(f"Preview download error: {e}")

    def generate_hash(self, file_path):
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""): hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def fetch_civitai_data_by_hash(self, file_path):
        try:
            file_hash = self.generate_hash(file_path)
            url = f"https://civitai.com/api/v1/model-versions/by-hash/{file_hash}"
            r = requests.get(url, headers=self.get_headers())
            if r.status_code != 200: return None, f"HTTP Error: {r.status_code}"
            data = r.json()
            if 'error' in data: return None, f"CivitAI Error: {data.get('error')}"
            return data, None
        except Exception as e: return None, str(e)

    def get_civitai_json_path(self, lora_full_path):
        rel_path = os.path.relpath(lora_full_path, self.lora_root)
        meta_dir = os.path.join(self.metadata_root, os.path.dirname(rel_path))
        if not os.path.exists(meta_dir): os.makedirs(meta_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(rel_path))[0]
        return os.path.join(meta_dir, base_name + ".json")

    def get_lset_path(self, lora_full_path):
        return os.path.splitext(lora_full_path)[0] + ".lset"

    def read_lset_data(self, lora_full_path):
        lset_path = self.get_lset_path(lora_full_path)
        if os.path.exists(lset_path):
            try:
                with open(lset_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except: pass
        return {}

    def read_lset_prompt_string(self, lora_full_path):
        data = self.read_lset_data(lora_full_path)
        raw = data.get("prompt", "")
        if isinstance(raw, list): return ", ".join(raw)
        return str(raw).strip()

    def write_lset(self, lora_full_path, prompt):
        lset_path = self.get_lset_path(lora_full_path)
        filename = os.path.basename(lora_full_path)

        existing = self.read_lset_data(lora_full_path)
        lora_list = existing.get("loras", [filename])
        
        data = {
            "loras": lora_list,
            "prompt": prompt
        }
        
        try:
            with open(lset_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            return True
        except Exception as e:
            print(f"Error writing lset: {e}")
            return False

    def _fetch_and_process_single_lora(self, full_path, key=None):
        data, err = self.fetch_civitai_data_by_hash(full_path)
        if err: return False, err

        dest = self.get_civitai_json_path(full_path)
        try:
            with open(dest, 'w', encoding='utf-8') as f: json.dump(data, f, indent=4)
        except Exception as e: return False, f"JSON save failed: {e}"

        mid = data.get('modelId')
        img_url = None
        if 'images' in data and data['images']:
            img_url = data['images'][0].get('url')
        elif mid:
             try:
                 r = requests.get(f"{REST_URL}/{mid}", headers=self.get_headers())
                 if r.status_code == 200:
                     mdata = r.json()
                     ver = next((v for v in mdata.get('modelVersions',[]) if v['id'] == data.get('id')), None)
                     if ver and ver.get('images'): img_url = ver['images'][0]['url']
                     elif mdata.get('modelVersions') and mdata['modelVersions'][0].get('images'):
                         img_url = mdata['modelVersions'][0]['images'][0]['url']
             except: pass
        
        if mid and img_url: self.download_preview_image(img_url, mid)

        trained_words = data.get('trainedWords', [])
        current_prompt = self.read_lset_prompt_string(full_path)
        
        updated_prompt = False
        if trained_words and not current_prompt:
            new_prompt = ", ".join(trained_words)
            if self.write_lset(full_path, new_prompt):
                updated_prompt = True

        msg = "Metadata updated."
        if updated_prompt: msg += " Default prompt set in .lset."
        return True, msg

    def resolve_target_folder(self, base_model_name):
        internal_key = CIVIT_TO_WANGP_ARCH.get(base_model_name)
        fallback_path = os.path.abspath("loras")
        if not internal_key: return fallback_path
        try:
            path = self.get_lora_dir(internal_key)
            if path and os.path.isdir(path): return os.path.abspath(path)
        except: pass
        return fallback_path

    def batch_update_metadata(self, state, category, current_files, progress=gr.Progress()):
        self.lora_root = self.discover_lora_root(state)
        files_to_process = []
        if category == "All LoRAs":
            for root, dirs, f_names in os.walk(self.lora_root):
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                rel_root = os.path.relpath(root, self.lora_root)
                if rel_root == ".": rel_root = ""
                for f in f_names:
                    if f.endswith(".safetensors") or f.endswith(".sft"):
                        path = os.path.join(rel_root, f) if rel_root else f
                        files_to_process.append(path)
        else:
            target_dir = os.path.join(self.lora_root, category)
            if os.path.isdir(target_dir):
                for f in os.listdir(target_dir):
                    if f.endswith(".safetensors") or f.endswith(".sft"): files.append(f)

        if not files_to_process:
            gr.Warning("No files found to update.")
            return 0

        updated_count = 0
        error_count = 0
        
        for i, item_name in enumerate(progress.tqdm(files_to_process, desc="Updating Metadata")):
            if category == "All LoRAs":
                full_path = os.path.join(self.lora_root, item_name)
            else:
                full_path = os.path.join(self.lora_root, category, item_name)

            if os.path.exists(full_path):
                success, msg = self._fetch_and_process_single_lora(full_path)
                if success: updated_count += 1
                else: error_count += 1
            time.sleep(0.1)

        gr.Info(f"Batch Update Complete. Updated: {updated_count}, Errors: {error_count}")
        return (self.refresh_trigger.value or 0) + 1

    def create_manager_ui(self):
        self.is_initialized = gr.State(False)
        self.refresh_trigger = gr.State(0)
        self.lora_selection_state = gr.State([])
        
        self.manager_to_browser_state = gr.State()

        gr.HTML("<style>.plugin-hidden-ui { display: none !important; }</style>")
        self.manager_to_browser_btn = gr.Button(visible=True, elem_id="manager_to_browser_btn", elem_classes=["plugin-hidden-ui"])
        self.lora_selection_bridge = gr.Textbox(elem_id="lora_selection_bridge", visible=True, elem_classes=["plugin-hidden-ui"])

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📂 Library")
                self.category_dropdown = gr.Dropdown(label="Category", choices=[], value=None, interactive=True)

                self.lora_html_list = gr.HTML(elem_id="lora_html_list")
                
                with gr.Row():
                    self.refresh_btn = gr.Button("🔄 Refresh", size="sm")
                    self.update_all_btn = gr.Button("🔄 Update Metadata", size="sm", variant="secondary")

                with gr.Accordion("⚙️ Settings", open=False):
                    self.api_key = gr.Textbox(
                        label="CivitAI API Key", 
                        type="password", 
                        value=self.saved_settings.get("api_key", ""),
                        info="Required for NSFW content and downloading certain models."
                    )
                    gr.Markdown(
                        '🔑 Get your API key from '
                        '[civitai.com/user/account](https://civitai.com/user/account) '
                        'by clicking **“Add API key”** under the **API Keys** section.'
                    )
                    self.auto_fetch_chk = gr.Checkbox(
                        label="Auto-fetch from CivitAI on select", 
                        value=self.saved_settings.get("auto_fetch", False)
                    )
                    self.preview_mode = gr.Radio(
                        label="Preview Download Mode",
                        choices=["Image Thumbnail (First Frame)", "Original Media (Video/Image)"],
                        value=self.saved_settings.get("preview_mode", "Image Thumbnail (First Frame)")
                    )
                    self.save_settings_btn = gr.Button("💾 Save Settings", size="sm")

            with gr.Column(scale=2):
                @gr.render(inputs=[self.lora_selection_state, self.refresh_trigger, self.auto_fetch_chk], triggers=[self.lora_selection_state.change, self.refresh_trigger.change])
                def render_lora_details(selected_items, trig_val, auto_fetch):
                    if not selected_items:
                        gr.Markdown("*Select a LoRA to view details.*")
                        return

                    gr.Markdown(f"### 📝 Details ({len(selected_items)} selected)")

                    for lora_name in selected_items:
                        full_path = self.resolve_path(lora_name)
                        
                        current_prompt_str = self.read_lset_prompt_string(full_path)
                        json_path = self.get_civitai_json_path(full_path)
                        
                        if not os.path.exists(json_path) and auto_fetch:
                            self._fetch_and_process_single_lora(full_path)
                            current_prompt_str = self.read_lset_prompt_string(full_path)

                        civitai_data = None
                        if os.path.exists(json_path):
                            try:
                                with open(json_path, 'r', encoding='utf-8') as f: civitai_data = json.load(f)
                            except: pass

                        with gr.Group():
                            with gr.Row():
                                gr.Markdown(f"#### {os.path.basename(lora_name)}")
                            
                            if civitai_data:
                                model_id = civitai_data.get('modelId')
                                images_list = civitai_data.get('images', [])
                                if images_list:
                                    img_urls = [img.get('url') for img in images_list[:4]]
                                    gr.Gallery(value=img_urls, label="CivitAI Gallery", columns=4, height=200, object_fit="contain")
                                else:
                                    gr.Markdown("*No remote images found.*")

                                model_name = civitai_data.get('model', {}).get('name', 'Unknown')
                                base_model = civitai_data.get('baseModel', 'Unknown')
                                gr.Markdown(f"**Model:** {model_name} | **Base:** {base_model}")
                                
                                with gr.Row():
                                    if model_id:
                                        view_btn = gr.Button("🔍 Browse on CivitAI", size="sm", variant="secondary")
                                        view_btn.click(fn=lambda m=model_id: m, inputs=None, outputs=[self.manager_to_browser_state]).then(fn=None, js="window.triggerManagerToBrowser")
                                    
                                    upd_btn = gr.Button("🔄 Update Info", size="sm")
                                    def do_upd(fp=full_path, cur_val=trig_val):
                                        self._fetch_and_process_single_lora(fp)
                                        return (cur_val or 0) + 1
                                    upd_btn.click(fn=do_upd, inputs=None, outputs=[self.refresh_trigger])

                                triggers = civitai_data.get('trainedWords', [])
                                if triggers: gr.Markdown(f"**Triggers:** `{', '.join(triggers)}`")

                            else:
                                gr.Markdown("*No metadata found.*")
                                man_fetch = gr.Button("🌐 Fetch Info", size="sm")
                                def do_fetch(fp=full_path, cur_val=trig_val):
                                    self._fetch_and_process_single_lora(fp)
                                    return (cur_val or 0) + 1
                                man_fetch.click(fn=do_fetch, inputs=None, outputs=[self.refresh_trigger])

                            path_state = gr.State(full_path)
                            prompt_input = gr.TextArea(value=current_prompt_str, label="Default Prompt", lines=2)
                            save_btn = gr.Button("💾 Save Prompt", size="sm")
                            save_btn.click(fn=self.save_lset_prompt, inputs=[path_state, prompt_input], outputs=None)
                        gr.Markdown("---")

                with gr.Column(visible=False) as self.actions_panel:
                    gr.Markdown("### Inject to Generator")
                    self.prompt_select_group = gr.CheckboxGroup(label="Select Prompts", choices=[])
                    self.lora_select_group = gr.CheckboxGroup(label="Select LoRAs", choices=[])
                    
                    with gr.Row():
                        self.prompt_mode = gr.Radio(["Append To Current Prompts", "Overwrite Current Prompts"], value="Append To Current Prompts", label="Prompt Injection Mode")
                        self.lora_mode = gr.Radio(["Append To Current Loras", "Overwrite Current Loras"], value="Append To Current Loras", label="Lora Injection Mode")
                    self.use_btn = gr.Button("✨ Send to Generator", variant="primary")

        self.on_tab_outputs = [self.is_initialized, self.category_dropdown, self.lora_html_list]

        self.category_dropdown.change(self.render_lora_grid, [self.state, self.category_dropdown, self.lora_selection_state], [self.lora_html_list])
        self.refresh_btn.click(self.ui_refresh_click, [self.state, self.category_dropdown, self.lora_selection_state], [self.category_dropdown, self.lora_html_list])
        self.update_all_btn.click(self.batch_update_metadata, [self.state, self.category_dropdown, self.lora_selection_state], [self.refresh_trigger])

        def save_manager_settings(key, auto, mode):
            self.save_settings_to_disk(api_key=key, auto_fetch=auto, preview_mode=mode)
            gr.Info("Settings saved!")

        self.save_settings_btn.click(
            save_manager_settings, 
            inputs=[self.api_key, self.auto_fetch_chk, self.preview_mode], 
            outputs=None
        )

        self.lora_selection_bridge.change(fn=lambda x: json.loads(x) if x else [], inputs=[self.lora_selection_bridge], outputs=[self.lora_selection_state])

        self.lora_selection_state.change(
            self.update_action_options, 
            [self.lora_selection_state], 
            [self.prompt_select_group, self.lora_select_group, self.actions_panel]
        )

        self.use_btn.click(
            self.finalize_injection, 
            [self.prompt_select_group, self.lora_select_group, self.prompt_mode, self.lora_mode, self.prompt, self.loras_choices], 
            [self.prompt, self.loras_choices, self.main_tabs]
        )

    def create_browser_ui(self):
        self.civit_items = gr.State([])
        self.civit_cursor = gr.State(None)
        self.civit_model_data = gr.State({})

        self.bridge_input = gr.Textbox(elem_id="civit_bridge_input", visible=True, elem_classes=["plugin-hidden-ui"])
        self.load_more_btn = gr.Button("Load More", elem_id="civit_load_more_btn", visible=True, elem_classes=["plugin-hidden-ui"])
        self.browser_bridge_trigger = gr.Button(visible=False) 

        d_sort = self.saved_settings.get("sort", "Highest Rated")
        d_period = self.saved_settings.get("period", "Week")
        d_nsfw = self.saved_settings.get("nsfw", True)
        d_types = self.saved_settings.get("types", ["Checkpoint", "LORA"])
        d_base = self.saved_settings.get("base", DEFAULT_BASE_SELECTION)

        with gr.Tabs() as self.civit_tabs:
            with gr.Tab("Browse", id="browse_tab"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=250):
                        self.query = gr.Textbox(label="Search", placeholder="Search models...")
                        with gr.Accordion("Filters", open=True):
                            self.sort = gr.Dropdown(SORT_OPTIONS, value=d_sort, label="Sort")
                            self.period = gr.Dropdown(PERIOD_OPTIONS, value=d_period, label="Period")
                            self.nsfw = gr.Checkbox(label="NSFW", value=d_nsfw)
                            self.types = gr.Dropdown(MODEL_TYPES, value=d_types, multiselect=True, label="Types")
                            self.base = gr.Dropdown(BASE_MODELS, value=d_base, multiselect=True, label="Base Models")
                        self.search_btn = gr.Button("Search / Browse", variant="primary")
                        self.status = gr.Markdown("Ready")
                    with gr.Column(scale=3):
                        self.html_results = gr.HTML()

            with gr.Tab("Details", id="details_tab"):
                self.back_btn = gr.Button("← Back to Browse")
                self.detail_header = gr.HTML()
                with gr.Row():
                    self.ver_dd = gr.Dropdown(label="Version", interactive=True)
                    self.file_dd = gr.Dropdown(label="File", interactive=True)
                
                self.target_folder = gr.Textbox(label="Target Folder / Category", value="loras", interactive=True)
                
                self.dl_btn = gr.Button("Download to Wan2GP", variant="primary")
                self.dl_status = gr.Textbox(label="Download Status", interactive=False)
                self.media_area = gr.HTML()

        self.manager_to_browser_btn.click(self.bridge_manager_to_browser, inputs=[self.manager_to_browser_state], outputs=[self.main_tabs, self.bridge_input, self.browser_bridge_trigger])

        def save_browser_settings(sort, period, nsfw, types, base):
            self.save_settings_to_disk(sort=sort, period=period, nsfw=nsfw, types=types, base=base)
        
        save_inputs = [self.sort, self.period, self.nsfw, self.types, self.base]
        for comp in save_inputs:
            comp.change(save_browser_settings, inputs=save_inputs, outputs=None)

        self.search_btn.click(self.run_search, [self.query, self.sort, self.period, self.types, self.base, self.nsfw, self.api_key], [self.html_results, self.civit_items, self.civit_cursor, self.status])
        self.load_more_btn.click(self.run_more, [self.query, self.sort, self.period, self.types, self.base, self.nsfw, self.api_key, self.civit_cursor, self.civit_items], [self.html_results, self.civit_items, self.civit_cursor, self.status])
        self.bridge_input.change(self.on_select_model, [self.bridge_input, self.civit_items, self.api_key], [self.civit_tabs, self.detail_header, self.ver_dd, self.civit_model_data, self.status, self.target_folder]).then(self.update_version_files, [self.ver_dd, self.civit_model_data], [self.file_dd, self.media_area, self.target_folder])
        self.ver_dd.change(self.update_version_files, [self.ver_dd, self.civit_model_data], [self.file_dd, self.media_area, self.target_folder])

        self.back_btn.click(lambda: gr.Tabs(selected="browse_tab"), None, self.civit_tabs)
        self.dl_btn.click(self.download_model, [self.file_dd, self.api_key, self.state, self.civit_model_data, self.target_folder], [self.dl_status])
        self.search_btn.click()

    def render_lora_grid(self, state, category, selected_list):
        self.lora_root = self.discover_lora_root(state)
        files = []
        if category == "All LoRAs":
            for root, dirs, f_names in os.walk(self.lora_root):
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                rel_root = os.path.relpath(root, self.lora_root)
                if rel_root == ".": rel_root = ""
                for f in f_names:
                    if f.endswith(".safetensors") or f.endswith(".sft"):
                        files.append(os.path.join(rel_root, f) if rel_root else f)
        elif category and os.path.isdir(os.path.join(self.lora_root, category)):
            target_dir = os.path.join(self.lora_root, category)
            for f in os.listdir(target_dir):
                if f.endswith(".safetensors") or f.endswith(".sft"):
                    files.append(f)
        files.sort()
        
        html = """<style>
            .lora-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 8px; max-height: 600px; overflow-y: auto; padding: 4px; }
            .lora-item { background: #252630; border-radius: 6px; overflow: hidden; border: 2px solid transparent; cursor: pointer; position: relative; transition: all 0.2s; aspect-ratio: 1; }
            .lora-item:hover { border-color: #4dabf7; }
            .lora-item.selected { border-color: #4dabf7; box-shadow: 0 0 0 2px rgba(77, 171, 247, 0.4); }
            .lora-thumb { width: 100%; height: 100%; object-fit: cover; background: #111; display: block; }
            .lora-name { position: absolute; bottom: 0; left: 0; width: 100%; background: rgba(0, 0, 0, 0.7); color: white !important; padding: 4px 6px; font-size: 11px; line-height: 1.2; font-weight: 600; text-shadow: none; z-index: 5; box-sizing: border-box; word-break: break-all; pointer-events: none; -webkit-font-smoothing: antialiased; }
            .lora-check { position: absolute; top: 4px; right: 4px; width: 18px; height: 18px; border-radius: 50%; background: #4dabf7; display: none; align-items: center; justify-content: center; color: white; font-size: 12px; font-weight: bold; z-index: 10; box-shadow: 0 2px 4px rgba(0,0,0,0.3); }
            .lora-item.selected .lora-check { display: flex; }
        </style><div class="lora-grid">"""
        
        for f in files:
            full_path = os.path.join(self.lora_root, f if category == "All LoRAs" else os.path.join(category, f))
            json_path = self.get_civitai_json_path(full_path)
            model_id = None
            if os.path.exists(json_path):
                try:
                    with open(json_path) as jf:
                        d = json.load(jf)
                        model_id = d.get('modelId')
                except: pass
            
            _, img_path = self.get_local_preview_path(model_id)
            is_video = img_path and (img_path.endswith('.mp4') or img_path.endswith('.webm'))
            img_content = ""
            if img_path:
                if is_video: img_content = f'<video src="{img_path}" class="lora-thumb" autoplay loop muted playsinline></video>'
                else: img_content = f'<img src="{img_path}" class="lora-thumb" loading="lazy">'
            else: img_content = f'<img src="{PLACEHOLDER_B64}" class="lora-thumb" loading="lazy">'
            
            is_sel = f in selected_list
            sel_class = "selected" if is_sel else ""
            f_esc = f.replace("'", "\\'")
            html += f"""<div class="lora-item {sel_class}" onclick="toggleLoraCard(this, '{f_esc}')"><div class="lora-check">✓</div>{img_content}<div class="lora-name">{os.path.basename(f)}</div></div>"""
        html += "</div>"
        return html

    def ui_refresh_click(self, state, sel, selected_list):
        _, dd, html = self.force_refresh(state, sel, selected_list)
        return dd, html

    def force_refresh(self, state, sel, selected_list=None):
        self.lora_root = self.discover_lora_root(state)
        dmap = self.build_category_map()
        folders = []
        if os.path.isdir(self.lora_root):
            for d in sorted([x for x in os.listdir(self.lora_root) if os.path.isdir(os.path.join(self.lora_root, x)) and not x.startswith('.')]):
                folders.append((dmap.get(d, d), d))
        choices = [("All LoRAs", "All LoRAs")] + folders
        val = sel if sel and sel in [c[1] for c in choices] else "All LoRAs"
        if val == "All LoRAs":
            try:
                td = self.get_lora_dir(self.get_state_model_type(state))
                tf = os.path.basename(td)
                if tf in [c[1] for c in choices]: val = tf
            except: pass
        return True, gr.update(choices=choices, value=val), self.render_lora_grid(state, val, selected_list or [])

    def run_search(self, q, s, p, t, b, n, k):
        items, nxt, msg = self.router_search(q, s, p, t, b, n, k, None)
        self.items_cache = items 
        html = self.render_html_grid(items)
        return html, items, nxt, msg

    def run_more(self, q, s, p, t, b, n, k, cur, existing):
        if not cur: return gr.update(), existing, cur, "No more pages"
        new_items, nxt, msg = self.router_search(q, s, p, t, b, n, k, cur)
        combined = existing + new_items
        self.items_cache = combined
        html = self.render_html_grid(combined)
        return html, combined, nxt, msg

    def router_search(self, query, sort, period, types, base, nsfw, key, cursor):
        if query and query.strip(): return self.search_rest(query, sort, period, nsfw, key, cursor)
        else: return self.browse_trpc(sort, period, types, base, nsfw, key, cursor)

    def search_rest(self, query, sort, period, nsfw, key, cursor):
        params = {"query": query, "limit": 20, "sort": sort, "period": period, "nsfw": "true" if nsfw else "false"}
        if cursor: params["cursor"] = cursor
        try:
            r = requests.get(REST_URL, headers=self.get_headers(key), params=params)
            r.raise_for_status()
            data = r.json()
            items = data.get("items", [])
            meta = data.get("metadata", {})
            next_cursor = meta.get("nextCursor") 
            if not next_cursor and meta.get("nextPage") and "cursor=" in meta["nextPage"]:
                try: next_cursor = meta["nextPage"].split("cursor=")[1].split("&")[0]
                except: pass
            return items, next_cursor, f"Search: {len(items)} results"
        except Exception as e: return [], None, f"Error: {e}"

    def browse_trpc(self, sort, period, types, base_models, nsfw, key, cursor):
        input_obj = {"json": {"period": period, "periodMode": "published", "sort": sort, "types": types if types else MODEL_TYPES, "baseModels": base_models if base_models else [], "browsingLevel": 31 if nsfw else 1, "cursor": cursor, "authed": bool(key)}, "meta": {"values": {"cursor": ["undefined"]}}}
        try:
            r = requests.get(TRPC_URL, headers=self.get_headers(key), params={"input": json.dumps(input_obj)})
            r.raise_for_status()
            data = r.json()
            json_data = data.get("result", {}).get("data", {}).get("json", {})
            return json_data.get("items", []), json_data.get("nextCursor"), f"Browse: {len(json_data.get('items', []))} items"
        except Exception as e: return [], None, f"Error: {e}"

    def render_html_grid(self, items):
        if not items:
            return "<div style='color:#888; padding:20px; text-align:center; font-size:1.2em;'>No models found.</div>"

        html = """<style>.civit-grid {display: grid;grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));gap: 16px;padding: 10px;font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;}.civit-card {background-color: #252630;border-radius: 12px;overflow: hidden;position: relative;padding-top: 150%;cursor: pointer;border: 1px solid #373a40;transition: transform 0.2s;box-sizing: border-box;}.civit-card:hover {transform: translateY(-4px);border-color: #4dabf7;box-shadow: 0 10px 20px rgba(0,0,0,0.5);}.civit-media-container {position: absolute;top: 0;left: 0;width: 100%;height: 100%;background: #101113;z-index: 1;}.civit-media {width: 100% !important;height: 100% !important;object-fit: cover !important;display: block;}.civit-overlay {position: absolute;bottom: 0;left: 0;right: 0;background: linear-gradient(to top, rgba(0,0,0,0.95) 0%, rgba(0,0,0,0.6) 60%, transparent 100%);padding: 50px 12px 12px 12px;pointer-events: none;z-index: 2;}.civit-title {color: #ffffff !important;font-weight: 700;font-size: 1rem;line-height: 1.3;text-shadow: 0 2px 3px rgba(0,0,0,1);margin-bottom: 6px;display: -webkit-box;-webkit-line-clamp: 3;-webkit-box-orient: vertical;overflow: hidden;}.civit-meta {display: flex;justify-content: space-between;align-items: center;font-size: 0.8rem;}.civit-badge {background: rgba(34, 139, 230, 0.9);color: #fff;padding: 3px 6px;border-radius: 4px;font-size: 0.75rem;font-weight: 500;}</style><div class="civit-grid">"""

        for item in items:
            mid = item.get('id')
            name = item.get('name', 'Unknown').replace('"', '&quot;')
            rank = item.get('rank', {}) or {}
            stats = item.get('stats', {}) or {}
            thumbs = rank.get('thumbsUpCount', stats.get('favoriteCount', stats.get('thumbsUpCount', 0))) or 0
            dls = rank.get('downloadCount', stats.get('downloadCount', 0)) or 0
            dls_str = f"{dls/1000:.1f}k" if dls > 1000 else str(dls)
            mtype = item.get('type', 'Model')
            media_html = ""
            poster_src = ""
            imgs = item.get('images', [])
            if not imgs and 'version' in item: imgs = item['version'].get('images', [])
            if not imgs and 'modelVersions' in item and item['modelVersions']: imgs = item['modelVersions'][0].get('images', [])

            if imgs:
                first = imgs[0]
                url = first.get('url')
                is_vid = first.get('type') == 'video' or (url and (url.endswith('.mp4') or url.endswith('.webm')))
                src = ""
                if url:
                    if "http" not in url:
                        src = self.construct_media_url(url, 450, is_vid)
                        if is_vid: poster_src = self.construct_media_url(url, 450, False)
                    else:
                        src = url
                        if is_vid: poster_src = src.replace('.mp4', '.jpg').replace('.webm', '.jpg')

                if is_vid:
                    media_html = f'<video class="civit-media" src="{src}" poster="{poster_src}" autoplay loop muted playsinline preload="auto"></video>'
                else:
                    _, local_url = self.get_local_preview_path(mid)
                    if local_url: media_html = f'<img class="civit-media" src="{local_url}" loading="lazy" alt="preview">'
                    else: media_html = f'<img class="civit-media" src="{src}" loading="lazy" alt="preview">'
            else:
                media_html = '<div class="civit-media" style="display:flex;align-items:center;justify-content:center;color:#666;background:#222;">No Preview</div>'

            html += f"""<div class="civit-card" onclick="window.civitSelectCard({mid})"><div class="civit-media-container">{media_html}</div><div class="civit-overlay"><div class="civit-title">{name}</div><div class="civit-meta"><span class="civit-badge">{mtype}</span><span style="color: #ffffff; text-shadow: 0 1px 3px rgba(0,0,0,1); font-weight: 600;">👍 {thumbs} · ⬇ {dls_str}</span></div></div></div>"""

        html += "</div>"
        return html

    def on_select_model(self, model_id_str, current_items, api_key):
        if not model_id_str:
            return gr.update(), "", gr.update(), {}, "Ready", gr.update(value="loras")
        
        try: mid = int(model_id_str)
        except: return gr.update(), "", gr.update(), {}, "Invalid ID", gr.update()

        preview = next((x for x in (current_items or []) if x.get('id') == mid), {})
        full_data = preview
        try:
            r = requests.get(f"{REST_URL}/{mid}", headers=self.get_headers(api_key))
            if r.status_code == 200: full_data = r.json()
        except: pass
        
        name = full_data.get('name', 'Unknown')
        creator = full_data.get('creator', {}).get('username', 'Unknown')
        desc = full_data.get('description', 'No description.')
        tags = ", ".join(full_data.get('tags', []))

        versions = full_data.get('modelVersions', [])
        if not versions and 'version' in full_data: versions = [full_data['version']]

        default_ver = versions[0] if versions else {}
        is_checkpoint = False

        for f in default_ver.get('files', []):
            if f.get('metadata', {}).get('size') == 'full':
                is_checkpoint = True
                break

        if not is_checkpoint and full_data.get('type') == 'Checkpoint':
            is_checkpoint = True

        if is_checkpoint:
            target_path = os.path.abspath("ckpts")
        else:
            base_model = default_ver.get('baseModel', 'Unknown')
            target_path = self.resolve_target_folder(base_model)

        info_html = f"""<style>.civit-details-box {{background-color: #1f2937; color: #ffffff !important; padding: 20px; border-radius: 12px; border: 1px solid #374151; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;}}.civit-details-box h1 {{ color: #ffffff !important; margin-top: 0; line-height: 1.2; font-size: 1.8em; }}.civit-details-box p, .civit-details-box li, .civit-details-box span, .civit-details-box div {{ color: #e5e7eb !important; line-height: 1.6;}}.civit-details-box a {{ color: #60a5fa !important; text-decoration: underline; }}.civit-details-box strong, .civit-details-box b {{ color: #ffffff !important; font-weight: 700; }}.civit-badge-prop {{background-color: #374151 !important;color: #ffffff !important;padding: 6px 10px;border-radius: 6px;display: inline-block;font-size: 0.9em;margin-right: 8px;margin-bottom: 8px;border: 1px solid #4b5563;}}</style><div class="civit-details-box"><h1>{name}</h1><div style="margin-bottom: 20px;"><span class="civit-badge-prop">🛠 <b>{creator}</b></span><span class="civit-badge-prop">📦 <b>{full_data.get('type')}</b></span><span class="civit-badge-prop">🧩 <b>{default_ver.get('baseModel', 'Unknown')}</b></span><span class="civit-badge-prop">🏷 {tags}</span></div><hr style="border-top: 1px solid #4a4d55; margin-bottom: 20px;"><div>{desc}</div></div>"""
        
        ver_choices = [(f"{v['name']} ({v.get('baseModel','?')})", v['id']) for v in versions]
        first_ver = ver_choices[0][1] if ver_choices else None
        
        return gr.Tabs(selected="details_tab"), info_html, gr.update(choices=ver_choices, value=first_ver), full_data, f"Loaded {name}", target_path

    def update_version_files(self, version_id, model_data):
        if not model_data or not version_id: return gr.update(choices=[]), "", gr.update()
        versions = model_data.get('modelVersions', [])
        if not versions and 'version' in model_data: versions = [model_data['version']]
        version = next((v for v in versions if v['id'] == version_id), None)
        if not version: return gr.update(choices=[]), "Version info missing", gr.update()
        is_checkpoint = False
        
        for f in version.get('files', []):
            if f.get('metadata', {}).get('size') == 'full':
                is_checkpoint = True
                break

        if not is_checkpoint and model_data.get('type') == 'Checkpoint':
             is_checkpoint = True

        if is_checkpoint:
            new_target_folder = os.path.abspath("ckpts")
        else:
            base_model = version.get('baseModel', 'Unknown')
            new_target_folder = self.resolve_target_folder(base_model)

        file_opts = []
        for f in version.get('files', []):
            label = f"{f.get('type','Model')} | {f['name']} | {round(f.get('sizeKB',0)/1024, 2)} MB"
            file_opts.append((label, f['downloadUrl']))

        media_html = "<div style='display:grid; grid-template-columns:repeat(auto-fill, minmax(250px, 1fr)); gap:15px;'>"
        for img in version.get('images', []):
            url = img.get('url')
            if not url: continue
            is_vid = img.get('type') == 'video' or url.endswith(('.mp4','.webm'))
            src = self.construct_media_url(url, 450, is_vid) if url and "http" not in url else url
            cell_style = "width:100%; aspect-ratio:2/3; background:#000; border-radius:8px; overflow:hidden; border:1px solid #333; position:relative;"
            media_style = "width:100%; height:100%; object-fit:contain; display:block;"
            if is_vid:
                poster = self.construct_media_url(url, 450, False) if url and "http" not in url else ""
                media_html += f"<div style='{cell_style}'><video src='{src}' poster='{poster}' loop muted playsinline controls onmouseover=\"this.play()\" onmouseout=\"this.pause()\" style='{media_style}'></video></div>"
            else:
                media_html += f"<div style='{cell_style}'><img src='{src}' loading='lazy' style='{media_style}'></div>"
        media_html += "</div>"
        
        return gr.update(choices=file_opts, value=file_opts[0][1] if file_opts else None), media_html, new_target_folder

    def download_model(self, url, key, state, model_data, target_dir_input):
        if not url: return "No file selected"
        is_checkpoint = False

        if model_data.get('type') == 'Checkpoint':
            is_checkpoint = True
        else:
            versions = model_data.get('modelVersions', [])
            found = False
            for v in versions:
                for f in v.get('files', []):
                    if f.get('downloadUrl') == url:
                        if f.get('metadata', {}).get('size') == 'full':
                            is_checkpoint = True
                        found = True
                        break
                if found: break

        if is_checkpoint:
            return self.create_finetune_definition(url, model_data, key, target_dir_input)

        target_dir = target_dir_input.strip()
        if not target_dir: target_dir = "loras"
        
        if not os.path.exists(target_dir):
            try:
                os.makedirs(target_dir, exist_ok=True)
            except Exception as e:
                return f"Error creating directory '{target_dir}': {e}"

        try:
            r = requests.get(url, headers=self.get_headers(key), stream=True)
            r.raise_for_status()

            fname = "model.safetensors"
            selected_version = None
            
            versions = model_data.get('modelVersions', [])
            for v in versions:
                for f in v.get('files', []):
                    if f['downloadUrl'] == url:
                        fname = f['name']
                        selected_version = v
                        break
                if selected_version: break

            if "content-disposition" in r.headers and "filename=" in r.headers["content-disposition"]:
                fname = r.headers["content-disposition"].split("filename=")[1].strip('"').strip(';')
            
            save_path = os.path.join(target_dir, fname)

            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(1024*1024): f.write(chunk)

            mid = model_data.get('id')
            if mid and selected_version:
                json_path = self.get_civitai_json_path(save_path)

                meta_payload = selected_version.copy()
                meta_payload['modelId'] = mid
                meta_payload['model'] = {
                    'name': model_data.get('name'),
                    'type': model_data.get('type'),
                    'nsfw': model_data.get('nsfw'),
                    'poi': model_data.get('poi')
                }
                
                try:
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(meta_payload, f, indent=4)
                except Exception as e:
                    print(f"Error saving metadata JSON: {e}")

                _, local_prev = self.get_local_preview_path(mid)
                if not local_prev:
                    images = selected_version.get('images', [])
                    if not images: images = model_data.get('images', [])
                    
                    if images:
                        self.download_preview_image(images[0]['url'], mid)

                triggers = selected_version.get('trainedWords', [])
                if triggers:
                    self.write_lset(save_path, ", ".join(triggers))

            return f"Saved to {save_path}"
        except Exception as e: 
            traceback.print_exc()
            return f"Error: {e}"

    def create_finetune_definition(self, download_url, model_data, api_key, target_dir=None):
        civit_base = "Unknown"
        versions = model_data.get('modelVersions', [])
        selected_version = None
        for v in versions:
            for f in v.get('files', []):
                if f['downloadUrl'] == download_url:
                    selected_version = v
                    break
            if selected_version: break
        
        if not selected_version and versions: selected_version = versions[0]
        if selected_version: civit_base = selected_version.get('baseModel', 'Unknown')
        wangp_arch = CIVIT_TO_WANGP_ARCH.get(civit_base)
        
        if not wangp_arch:
            return f"Error: Could not map CivitAI Base Model '{civit_base}' to a WanGP Architecture. Manual download required."

        ckpt_dir = target_dir if target_dir else os.path.abspath("ckpts")
        os.makedirs(ckpt_dir, exist_ok=True)

        try:
            r = requests.get(download_url, headers=self.get_headers(api_key), stream=True)
            r.raise_for_status()
            
            fname = "model.safetensors"
            if "content-disposition" in r.headers and "filename=" in r.headers["content-disposition"]:
                fname = r.headers["content-disposition"].split("filename=")[1].strip('"').strip(';')
            else:
                if selected_version:
                    for f in selected_version.get('files', []):
                        if f['downloadUrl'] == download_url:
                            fname = f['name']
                            break
            
            ckpt_path = os.path.join(ckpt_dir, fname)

            with open(ckpt_path, 'wb') as f:
                for chunk in r.iter_content(1024*1024): 
                    if chunk: f.write(chunk)
                    
        except Exception as e:
            return f"Error downloading checkpoint: {e}"

        safe_name = "".join([c for c in model_data.get('name', 'Unknown') if c.isalnum() or c in (' ', '-', '_')]).strip()
        filename = safe_name.replace(" ", "_") + ".json"
        save_path = os.path.join(self.finetunes_root, filename)
        
        trained_words = selected_version.get('trainedWords', []) if selected_version else []
        prompt_str = ", ".join(trained_words) if trained_words else ""
        description = f"Imported from CivitAI. Base: {civit_base}. {model_data.get('description', '')[:200]}..."
        
        finetune_data = {
            "settings_version": 2.41,
            "prompt": prompt_str,
            "model": {
                "name": model_data.get('name', 'Unknown'),
                "architecture": wangp_arch,
                "description": description,
                "URLs": [fname],
                "auto_quantize": True
            }
        }
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f: json.dump(finetune_data, f, indent=4)
            mid = model_data.get('id')
            if mid and model_data.get('images'): self.download_preview_image(model_data['images'][0]['url'], mid)
            return f"Downloaded checkpoint to {ckpt_path} and created Finetune Definition: {filename}. Restart WanGP to see it."
        except Exception as e: return f"Error creating finetune definition: {e}"

    def bridge_manager_to_browser(self, mid):
        if not mid: return gr.update(), gr.update(), gr.update()
        return gr.Tabs(selected="plugin_civitai_browser_tab"), str(mid), None

    def on_tab_select(self, state):
        if not self.category_dropdown.choices:
            _, category_update, lora_list_update = self.force_refresh(state, None, None)
            return gr.update(value=True), category_update, lora_list_update
        return gr.update(value=True), gr.update(), gr.update()

    def resolve_path(self, item_name):
        is_recursive = os.path.sep in item_name or "/" in item_name
        if is_recursive: return os.path.join(self.lora_root, item_name)
        for root, _, files in os.walk(self.lora_root):
            if item_name in files: return os.path.join(root, item_name)
        return item_name

    def save_lset_prompt(self, full_path, prompt):
        if not full_path or not os.path.exists(full_path): return
        self.write_lset(full_path, prompt)
        gr.Info("Saved to .lset file!")

    def discover_lora_root(self, state):
        try: 
            d = self.get_lora_dir(self.get_state_model_type(state))
            if d and os.path.isdir(d): return os.path.dirname(d)
        except: pass
        return "loras"

    def build_category_map(self):
        folder_to_models = {}
        if hasattr(self, 'model_types') and self.model_types:
            for mtype in self.model_types:
                try:
                    path = self.get_lora_dir(mtype)
                    if path:
                        folder = os.path.basename(path)
                        dummy = [""]
                        pname = self.get_model_name(mtype, dummy)
                        if folder not in folder_to_models: folder_to_models[folder] = []
                        if pname not in folder_to_models[folder]: folder_to_models[folder].append(pname)
                except: continue
        dmap = {}
        for f, ms in folder_to_models.items():
            if not ms: dmap[f] = f
            else:
                mstr = ", ".join(ms[:2]) + (", ..." if len(ms)>2 else "")
                dmap[f] = f"{f} ({mstr})"
        return dmap

    def update_action_options(self, selected_items):
        if not selected_items:
            return gr.update(choices=[], value=[], visible=False), gr.update(choices=[], value=[], visible=False), gr.update(visible=False)
        
        all_prompts = []
        all_loras = set()
        
        for lora in selected_items:
            all_loras.add(os.path.basename(lora))

            full_path = self.resolve_path(lora)
            data = self.read_lset_data(full_path)

            p = data.get("prompt", "")
            if p:
                if isinstance(p, list): all_prompts.extend([x.strip() for x in p if x.strip()])
                elif isinstance(p, str) and p.strip(): all_prompts.append(p.strip())

            deps = data.get("loras", [])
            for d in deps:
                all_loras.add(os.path.basename(d))
        
        unique_prompts = sorted(list(set(all_prompts)))
        unique_loras = sorted(list(all_loras))
        
        return (
            gr.update(choices=unique_prompts, value=unique_prompts, visible=True), 
            gr.update(choices=unique_loras, value=unique_loras, visible=True),
            gr.update(visible=True)
        )

    def finalize_injection(self, selected_prompts, selected_loras, prompt_mode, lora_mode, current_prompt, current_loras):
        p_text = ", ".join(selected_prompts) if selected_prompts else ""
        new_prompt = p_text if prompt_mode == "Overwrite Current Prompts" else (f"{current_prompt}\n{p_text}" if current_prompt and p_text else (current_prompt or p_text))

        final_loras = [] if lora_mode == "Overwrite Current Loras" else (current_loras or []).copy()
        
        if selected_loras:
            for l in selected_loras:
                if l not in final_loras:
                    final_loras.append(l)
            
        gr.Info(f"Injected {len(selected_loras) if selected_loras else 0} LoRAs and {len(selected_prompts) if selected_prompts else 0} prompts")
        return new_prompt, final_loras, self.goto_video_tab(None)
