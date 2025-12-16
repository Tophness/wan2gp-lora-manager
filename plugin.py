import gradio as gr
import os
import json
from shared.utils.plugins import WAN2GPPlugin

class LoraManagerPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "LoRA Manager"
        self.version = "1.0.0"
        self.description = "Multi-LoRA management with conflict resolution controls."
        self.lora_root = "loras" 

        self.plugin_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(self.plugin_dir, "lora_db.json")
        self.lora_metadata = {}

    def setup_ui(self):
        self.request_global("get_lora_dir")
        self.request_global("get_state_model_type")
        self.request_global("model_types") 
        self.request_global("get_model_name") 

        self.request_component("state")
        self.request_component("prompt") 
        self.request_component("loras_choices")
        self.request_component("main_tabs")

        self.load_json_db()
        self.on_tab_outputs = [] 

        self.add_tab(
            tab_id="lora_manager_tab",
            label="LoRA Manager",
            component_constructor=self.create_manager_ui,
            position=2
        )

    def create_manager_ui(self):
        self.is_initialized = gr.State(False)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📂 Library")
                
                self.category_dropdown = gr.Dropdown(
                    label="Category",
                    choices=[],
                    value=None,
                    interactive=True
                )
                
                self.lora_list = gr.CheckboxGroup(
                    choices=[],
                    label="Available LoRAs",
                    info="Select LoRAs to view details and inject.",
                    interactive=True,
                    elem_classes="lora-checkbox-list"
                )
                self.refresh_btn = gr.Button("🔄 Refresh List", size="sm")

            with gr.Column(scale=2):

                @gr.render(inputs=self.lora_list, triggers=[self.lora_list.change])
                def render_lora_cards(selected_items):
                    if not selected_items:
                        gr.Markdown("### 📝 Details")
                        gr.Markdown("*Select a LoRA from the list on the left to view details and edit prompts.*")
                        return

                    gr.Markdown(f"### 📝 Selected Details ({len(selected_items)})")
                    
                    for lora_name in selected_items:
                        key, full_path = self.resolve_path(lora_name)
                        current_prompt = self.lora_metadata.get(key, {}).get("prompt", "")
                        
                        with gr.Group():
                            with gr.Row(elem_classes="lora-card-header"):
                                gr.Markdown(f"#### 🏷️ {os.path.basename(lora_name)}")
                            
                            if "All LoRAs" in str(self.category_dropdown.value):
                                gr.Markdown(f"*(Folder: {os.path.dirname(lora_name)})*")

                            key_state = gr.State(key)
                            
                            prompt_input = gr.TextArea(
                                value=current_prompt,
                                label="Default Trigger / Prompt",
                                placeholder="Trigger words...",
                                lines=2,
                                interactive=True
                            )
                            
                            save_btn = gr.Button("Save", size="sm")
                            
                            save_btn.click(
                                fn=self.save_metadata,
                                inputs=[key_state, prompt_input],
                                outputs=None
                            )
                        gr.Markdown("---")

                with gr.Column(visible=False) as self.actions_panel:
                    gr.Markdown("### ⚙️ Injection Settings")
                    with gr.Row():
                        self.prompt_mode = gr.Radio(
                            choices=["Append", "Overwrite"],
                            value="Append",
                            label="Prompt Mode",
                            interactive=True
                        )
                        self.lora_mode = gr.Radio(
                            choices=["Append", "Overwrite"],
                            value="Append",
                            label="LoRA List Mode",
                            interactive=True
                        )
                    
                    self.use_btn = gr.Button("✨ Send to Generator", variant="primary")

                    with gr.Column(visible=False) as self.conflict_panel:
                        gr.Markdown("---")
                        gr.Markdown("#### ⚠️ Conflict Resolution")
                        gr.Markdown("Multiple prompts detected. How should they be handled?")
                        self.prompt_choice = gr.Radio(
                            choices=[],
                            label="Choose Strategy",
                            interactive=True
                        )
                        with gr.Row():
                            self.confirm_inject_btn = gr.Button("Confirm", variant="stop")
                            self.cancel_inject_btn = gr.Button("Cancel", variant="secondary")

        self.on_tab_outputs = [self.is_initialized, self.category_dropdown, self.lora_list]

        def toggle_actions(selected):
            return gr.update(visible=bool(selected))

        def reset_conflict_panel():
            return gr.update(visible=False), gr.update(value=None)

        self.lora_list.change(
            fn=toggle_actions,
            inputs=[self.lora_list],
            outputs=[self.actions_panel]
        ).then(
            fn=reset_conflict_panel,
            inputs=None,
            outputs=[self.conflict_panel, self.prompt_choice]
        )

        self.category_dropdown.change(
            fn=self.update_list_by_category,
            inputs=[self.category_dropdown],
            outputs=[self.lora_list]
        )

        self.refresh_btn.click(
            fn=self.force_refresh,
            inputs=[self.state, self.category_dropdown],
            outputs=[self.category_dropdown, self.lora_list]
        )

        self.use_btn.click(
            fn=self.prepare_injection,
            inputs=[self.lora_list, self.prompt_mode, self.lora_mode],
            outputs=[self.conflict_panel, self.prompt_choice, self.prompt, self.loras_choices, self.main_tabs]
        )

        self.confirm_inject_btn.click(
            fn=self.finalize_injection,
            inputs=[self.lora_list, self.prompt_choice, self.prompt_mode, self.lora_mode, self.prompt, self.loras_choices],
            outputs=[self.prompt, self.loras_choices, self.main_tabs, self.conflict_panel]
        )

        self.cancel_inject_btn.click(
            fn=reset_conflict_panel,
            inputs=None,
            outputs=[self.conflict_panel, self.prompt_choice]
        )

    def on_tab_select(self, state):
        return self.handle_tab_load(state)

    def handle_tab_load(self, state):
        if getattr(self, 'has_loaded_once', False):
            return gr.update(), gr.update(), gr.update()
        
        self.has_loaded_once = True
        is_init, dd_update, list_update = self.force_refresh(state, None)
        return is_init, dd_update, list_update

    def load_json_db(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    self.lora_metadata = json.load(f)
            except:
                self.lora_metadata = {}
        else:
            self.lora_metadata = {}

    def resolve_path(self, item_name):
        is_recursive = os.path.sep in item_name or "/" in item_name
        
        if is_recursive:
            full_path = os.path.join(self.lora_root, item_name)
            key = item_name
        else:
            full_path = ""
            key = ""
            for root, _, files in os.walk(self.lora_root):
                if item_name in files:
                    full_path = os.path.join(root, item_name)
                    key = os.path.join(os.path.relpath(root, self.lora_root), item_name)
                    break
        
        key = key.replace("\\", "/") 
        return key, full_path

    def save_metadata(self, key, prompt):
        if not key: return
        key = key.replace("\\", "/") 
        
        if key not in self.lora_metadata: self.lora_metadata[key] = {}
        self.lora_metadata[key]["prompt"] = prompt
        
        try:
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(self.lora_metadata, f, indent=4)
            gr.Info(f"Saved prompt!")
        except Exception as e:
            gr.Error(f"Save error: {e}")

    def discover_lora_root(self, state):
        model_type = self.get_state_model_type(state)
        try:
            specific_dir = self.get_lora_dir(model_type)
            if specific_dir and os.path.isdir(specific_dir):
                return os.path.dirname(specific_dir)
        except:
            pass
        return "loras"

    def build_category_map(self):
        folder_to_models = {}
        
        if hasattr(self, 'model_types') and self.model_types:
            for mtype in self.model_types:
                try:
                    path = self.get_lora_dir(mtype)
                    if path:
                        folder = os.path.basename(path)
                        dummy_list = [""]
                        pretty_name = self.get_model_name(mtype, dummy_list)
                        
                        if folder not in folder_to_models:
                            folder_to_models[folder] = []
                        
                        if pretty_name not in folder_to_models[folder]:
                            folder_to_models[folder].append(pretty_name)
                except:
                    continue
        
        display_map = {}
        for folder, models in folder_to_models.items():
            if not models:
                display_map[folder] = folder
            else:
                model_str = ", ".join(models[:2])
                if len(models) > 2: model_str += ", ..."
                display_map[folder] = f"{folder} ({model_str})"
        
        return display_map

    def force_refresh(self, state, current_selection):
        self.lora_root = self.discover_lora_root(state)
        display_map = self.build_category_map()
        
        folder_choices = [] 
        if os.path.isdir(self.lora_root):
            subdirs = [
                d for d in os.listdir(self.lora_root) 
                if os.path.isdir(os.path.join(self.lora_root, d)) 
                and not d.startswith('.') and not d.startswith('__')
            ]
            
            for d in sorted(subdirs):
                label = display_map.get(d, d)
                folder_choices.append((label, d))
        
        choices = [("All LoRAs", "All LoRAs")] + folder_choices
        
        selected_val = current_selection
        valid_values = [c[1] for c in choices]
        
        if not selected_val or selected_val not in valid_values:
            current_model_type = self.get_state_model_type(state)
            try:
                target_dir = self.get_lora_dir(current_model_type)
                target_folder = os.path.basename(target_dir)
                
                if target_folder in valid_values:
                    selected_val = target_folder
                else:
                    selected_val = "All LoRAs"
            except:
                selected_val = "All LoRAs"

        list_update = self.update_list_by_category(selected_val)
        return True, gr.update(choices=choices, value=selected_val), list_update

    def update_list_by_category(self, category):
        files = []
        if not category or not os.path.isdir(self.lora_root):
            return gr.update(choices=[])

        if category == "All LoRAs":
            for root, dirs, f_names in os.walk(self.lora_root):
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                rel_root = os.path.relpath(root, self.lora_root)
                if rel_root == ".": rel_root = ""
                
                for f in f_names:
                    if f.endswith(".safetensors") or f.endswith(".sft"):
                        if rel_root:
                            files.append(os.path.join(rel_root, f))
                        else:
                            files.append(f)
        else:
            target_dir = os.path.join(self.lora_root, category)
            if os.path.isdir(target_dir):
                for f in os.listdir(target_dir):
                    if f.endswith(".safetensors") or f.endswith(".sft"):
                        files.append(f)
        
        files.sort()
        return gr.update(choices=files, label=f"Files in {category}")

    def prepare_injection(self, selected_loras, prompt_mode, lora_mode):
        if not selected_loras:
            gr.Warning("No LoRAs selected.")
            return gr.update(visible=False), gr.update(), gr.update(), gr.update(), gr.update()

        prompts = []
        for l in selected_loras:
            key, _ = self.resolve_path(l)
            p = self.lora_metadata.get(key, {}).get("prompt", "")
            if p: prompts.append((os.path.basename(l), p))

        if len(selected_loras) == 1:
            p_text = prompts[0][1] if prompts else ""
            new_prompt, new_choices, tab_upd = self._perform_inject(selected_loras, p_text, prompt_mode, lora_mode)
            return gr.update(visible=False), gr.update(), new_prompt, new_choices, tab_upd

        if not prompts:
             new_prompt, new_choices, tab_upd = self._perform_inject(selected_loras, "", prompt_mode, lora_mode)
             return gr.update(visible=False), gr.update(), new_prompt, new_choices, tab_upd

        choices = []
        combined = []
        for name, p in prompts:
            choices.append((f"Use {name} prompt only", p))
            if p: combined.append(p)
        
        combined_str = ", ".join(combined)
        if combined_str:
            choices.append(("Combine All Prompts", combined_str))
        
        choices.append(("Don't add prompt (LoRAs only)", ""))
        
        return (
            gr.update(visible=True),
            gr.update(choices=choices, value=combined_str if combined_str else ""),
            gr.update(), gr.update(), gr.update()
        )

    def finalize_injection(self, selected_loras, prompt_choice, prompt_mode, lora_mode, current_prompt, current_loras):
        new_prompt, new_choices, tab_update = self._perform_inject(
            selected_loras, 
            prompt_choice, 
            prompt_mode,
            lora_mode,
            current_prompt,
            current_loras
        )
        return new_prompt, new_choices, tab_update, gr.update(visible=False)

    def _perform_inject(self, selected_loras_list, prompt_text, prompt_mode, lora_mode, current_ui_prompt="", current_ui_loras=None):
        if prompt_mode == "Overwrite":
            new_prompt = prompt_text
        else:
            new_prompt = current_ui_prompt or ""
            if prompt_text:
                if new_prompt:
                    new_prompt += "\n" + prompt_text
                else:
                    new_prompt = prompt_text

        if current_ui_loras is None: current_ui_loras = []
        if not isinstance(current_ui_loras, list): current_ui_loras = []
        
        if lora_mode == "Overwrite":
            final_loras = []
        else:
            final_loras = current_ui_loras.copy()
        
        for l in selected_loras_list:
            base = os.path.basename(l)
            if base not in final_loras:
                final_loras.append(base)

        gr.Info(f"Injected {len(selected_loras_list)} LoRAs")
        return new_prompt, final_loras, gr.Tabs(selected="video_gen")