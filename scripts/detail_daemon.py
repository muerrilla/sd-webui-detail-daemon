import os
import gradio as gr
import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import modules.scripts as scripts
from modules.script_callbacks import on_cfg_denoiser, remove_callbacks_for_function, on_infotext_pasted, on_ui_settings
from modules.ui_components import InputAccordion
from modules import shared

try:
    import modules_forge.forge_version
    is_forge = True
except:
    is_forge = False

def add_settings():
    section = ('detail_daemon', "Detail Daemon")
    shared.opts.add_option("detail_daemon_count", shared.OptionInfo(
        6, "Daemon count", gr.Slider, {"minimum": 1, "maximum": 10, "step": 1}, section=section).needs_reload_ui())
    shared.opts.add_option("detail_daemon_verbose", shared.OptionInfo(
        False, "Verbose", gr.Checkbox, {"interactive": True}, section=section))

on_ui_settings(add_settings)

def parse_infotext(infotext, params):
    try:
        raw = params['Detail Daemon']
        if raw.startswith("D"):  
            daemons = raw.split(";")
            if len(daemons) > shared.opts.data.get("detail_daemon_count", 6):
                tqdm.write(f"\033[31mDetail Daemon:\033[0m Need more daemons! Go to 'Settings > Uncategorized > Detail Daemon' and increase count to {len(daemons)}.")
            dd_dict = {}
            for idx, daemon in enumerate(daemons):
                tag, values = daemon.split(":")
                vals = values.split(",")
                dd_dict[f"active{idx + 1 if idx > 0 else ''}"] = bool(int(vals[0]))
                dd_dict[f"hr{idx + 1 if idx > 0 else ''}"] = bool(int(vals[1]))
                dd_dict[f"mode{idx + 1 if idx > 0 else ''}"] = vals[2]
                dd_dict[f"amount{idx + 1 if idx > 0 else ''}"] = float(vals[3])
                dd_dict[f"st{idx + 1 if idx > 0 else ''}"] = float(vals[4])
                dd_dict[f"ed{idx + 1 if idx > 0 else ''}"] = float(vals[5])
                dd_dict[f"bias{idx + 1 if idx > 0 else ''}"] = float(vals[6])
                dd_dict[f"exp{idx + 1 if idx > 0 else ''}"] = float(vals[7])
                dd_dict[f"st_offset{idx + 1 if idx > 0 else ''}"] = float(vals[8])
                dd_dict[f"ed_offset{idx + 1 if idx > 0 else ''}"] = float(vals[9])
                dd_dict[f"fade{idx + 1 if idx > 0 else ''}"] = float(vals[10])
                dd_dict[f"smooth{idx + 1 if idx > 0 else ''}"] = bool(int(vals[11]))
            params['Detail Daemon'] = dd_dict
        else:
            # fallback to old format for backward compatibility
            d = {}
            for s in raw.split(','):
                k, _, v = s.partition(':')
                d[k.strip()] = v.strip()
            params['Detail Daemon'] = d
    except Exception:
        pass

on_infotext_pasted(parse_infotext)


class Script(scripts.Script):

    def __init__(self):
        self.tab_param_count = 0 

    def title(self):
        return "Detail Daemon"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):

        def extract_infotext(d: dict, key, old_key=None):
            if 'Detail Daemon' in d:
                return d['Detail Daemon'].get(key)
            return d.get(old_key)

        daemon_count = shared.opts.data.get("detail_daemon_count", 6)
        all_params = []
        
        with InputAccordion(False, label="Detail Daemon", elem_id=self.elem_id('detail-daemon')) as gr_enabled:
            all_params.append(gr_enabled)
            self.infotext_fields = [(gr_enabled, lambda d: 'Detail Daemon' in d or 'DD_enabled' in d)]
            thumbs = []
            with gr.Group():
                with gr.Row(elem_classes=['detail-daemon-thumb-group']):
                    for i in range(daemon_count):                
                        _, empty = self.visualize(False, 0, 1, 0.5, 0, 0, 0, 0, 0, False, 'both', False) 
                        gr_thumb = gr.Plot(value=empty, elem_classes=['detail-daemon-thumb'], show_label=False)
                        thumbs.append(gr_thumb)
            with gr.Group(elem_classes=['detail-daemon-tab-group']):
                for i in range(daemon_count):
                    with gr.Tab(f'{["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"][i]}', elem_classes=['detail-daemon-tab']):
                        params_set = []
                        with gr.Row():
                            gr_active = gr.Checkbox(label="Active", value=False, min_width=60, elem_classes=['detail-daemon-active']) 
                            gr_hires = gr.Checkbox(label="Hires Pass", value=False, min_width=60, elem_classes=['detail-daemon-hires']) 
                        with gr.Row():
                            with gr.Column(scale=2, elem_classes=['detail-daemon-params']):                    
                                gr_amount_slider = gr.Slider(minimum=-1.00, maximum=1.00, step=.01, value=0.10, label="Detail Amount")
                                with gr.Row(): 
                                    gr_start = gr.Slider(minimum=0.0, maximum=1.0, step=.01, value=0.2, label="Start", min_width=60)
                                    gr_end = gr.Slider(minimum=0.0, maximum=1.0, step=.01, value=0.8, label="End", min_width=60) 
                                with gr.Row(): 
                                    gr_start_offset_slider = gr.Slider(minimum=-1.00, maximum=1.00, step=.01, value=0.00, label="Start Offset", min_width=60) 
                                    gr_end_offset_slider = gr.Slider(minimum=-1.00, maximum=1.00, step=.01, value=0.00, label="End Offset", min_width=60) 
                                with gr.Row(): 
                                    gr_bias = gr.Slider(minimum=0.0, maximum=1.0, step=.01, value=0.5, label="Bias", min_width=60)
                                    gr_exponent = gr.Slider(minimum=0.0, maximum=10.0, step=.05, value=1.0, label="Exponent", min_width=60)
                                gr_fade = gr.Slider(minimum=0.0, maximum=1.0, step=.05, value=0.0, label="Fade", min_width=60)
                            with gr.Column(scale=1, min_width=275):  
                                preview, _ = self.visualize(False, 0.2, 0.8, 0.5, 0.1, 1, 0, 0, 0, True, 'both', False)                                 
                                gr_vis = gr.Plot(value=preview, elem_classes=['detail-daemon-vis'], show_label=False)
                                gr_smooth = gr.Checkbox(label="Smooth", value=True, min_width=60, elem_classes=['detail-daemon-smooth'])
                        with gr.Accordion("More Knobs:", elem_classes=['detail-daemon-more-accordion'], open=False):
                            with gr.Row():
                                with gr.Column(scale=2):                                           
                                    with gr.Row():
                                        # Because the slider max and min are sometimes too limiting:
                                        gr_amount = gr.Number(value=0.10, precision=4, step=.01, label="Amount", min_width=60)  
                                        gr_start_offset = gr.Number(value=0.0, precision=4, step=.01, label="Start Offset", min_width=60)  
                                        gr_end_offset = gr.Number(value=0.0, precision=4, step=.01, label="End Offset", min_width=60) 
                                        gr_mode = gr.Dropdown(["both", "cond", "uncond"], value="both", label="Mode", show_label=True, min_width=60, elem_classes=['detail-daemon-mode']) 

                        gr_amount_slider.release(None, gr_amount_slider, gr_amount, _js="(x) => x")
                        gr_amount.change(None, gr_amount, gr_amount_slider, _js="(x) => x")

                        gr_start_offset_slider.release(None, gr_start_offset_slider, gr_start_offset, _js="(x) => x")
                        gr_start_offset.change(None, gr_start_offset, gr_start_offset_slider, _js="(x) => x")

                        gr_end_offset_slider.release(None, gr_end_offset_slider, gr_end_offset, _js="(x) => x")
                        gr_end_offset.change(None, gr_end_offset, gr_end_offset_slider, _js="(x) => x")

                        vis_args = [gr_active, gr_start, gr_end, gr_bias, gr_amount, gr_exponent, gr_start_offset, gr_end_offset, gr_fade, gr_smooth, gr_mode, gr_hires]
                        for vis_arg in vis_args:
                            if isinstance(vis_arg, gr.components.Slider):
                                vis_arg.release(fn=self.visualize, show_progress=False, inputs=vis_args, outputs=[gr_vis, thumbs[i]])
                            else:
                                vis_arg.change(fn=self.visualize, show_progress=False, inputs=vis_args, outputs=[gr_vis, thumbs[i]])

                        params_set = [
                            gr_active, gr_hires, gr_mode, gr_start, gr_end, gr_bias, gr_amount,
                            gr_exponent, gr_start_offset, gr_end_offset, gr_fade, gr_smooth
                        ]
                        all_params.extend(params_set)

                        # First tab should be backward compatible with the older single daemon Detail Daemon, hence the variable names without numbers
                        # older single daemon DD was backward compatible with yet older DD which had infotext with the DD_stuff format, so that's handled here too
                        if i == 0 :
                            self.tab_param_count = len(params_set)
                            self.infotext_fields.extend([                        
                                (gr_active, lambda d: extract_infotext(d, 'active') or 'Detail Daemon' in d or 'DD_enabled' in d),
                                (gr_hires, lambda d: extract_infotext(d, 'hr') or False),
                                (gr_mode, lambda d: extract_infotext(d, 'mode', 'DD_mode')),
                                (gr_amount, lambda d: extract_infotext(d, 'amount', 'DD_amount')),
                                (gr_start, lambda d: extract_infotext(d, 'st', 'DD_start')),
                                (gr_end, lambda d: extract_infotext(d, 'ed', 'DD_end')),
                                (gr_bias, lambda d: extract_infotext(d, 'bias', 'DD_bias')),
                                (gr_exponent, lambda d: extract_infotext(d, 'exp', 'DD_exponent')),
                                (gr_start_offset, lambda d: extract_infotext(d, 'st_offset', 'DD_start_offset')),
                                (gr_end_offset, lambda d: extract_infotext(d, 'ed_offset', 'DD_end_offset')),
                                (gr_fade, lambda d: extract_infotext(d, 'fade', 'DD_fade')),
                                (gr_smooth, lambda d: extract_infotext(d, 'smooth', 'DD_smooth')),
                            ])
                        else:                        
                            tab_tag = i + 1 
                            self.infotext_fields.extend([
                                (gr_active, lambda d, key=f'active{tab_tag}': extract_infotext(d, key) or False),
                                (gr_hires, lambda d, key=f'hr{tab_tag}': extract_infotext(d, key)),
                                (gr_mode, lambda d, key=f'mode{tab_tag}': extract_infotext(d, key)),
                                (gr_amount, lambda d, key=f'amount{tab_tag}': extract_infotext(d, key)),
                                (gr_start, lambda d, key=f'st{tab_tag}': extract_infotext(d, key)),
                                (gr_end, lambda d, key=f'ed{tab_tag}': extract_infotext(d, key)),
                                (gr_bias, lambda d, key=f'bias{tab_tag}': extract_infotext(d, key)),
                                (gr_exponent, lambda d, key=f'exp{tab_tag}': extract_infotext(d, key)),
                                (gr_start_offset, lambda d, key=f'st_offset{tab_tag}': extract_infotext(d, key)),
                                (gr_end_offset, lambda d, key=f'ed_offset{tab_tag}': extract_infotext(d, key)),
                                (gr_fade, lambda d, key=f'fade{tab_tag}': extract_infotext(d, key)),
                                (gr_smooth, lambda d, key=f'smooth{tab_tag}': extract_infotext(d, key)),
                            ])
        return all_params
    
    def process(self, p, enabled, *all_daemon_args):    
        if not enabled:
            if hasattr(self, 'callback_added'):
                remove_callbacks_for_function(self.denoiser_callback)
                delattr(self, 'callback_added')
            return

        if p.sampler_name in ["DPM adaptive", "HeunPP2"]:
            tqdm.write(f'\033[31mDetail Daemon:\033[0m Selected sampler ({p.sampler_name}) is not supported.')
            return        

        self.daemon_data = []
        extra_gen_texts = []        
        num_daemons = len(all_daemon_args) // self.tab_param_count

        for i in range(num_daemons):
            start_idx = i * self.tab_param_count
            end_idx = start_idx + self.tab_param_count
            daemon_args = all_daemon_args[start_idx:end_idx]

            active, hires, mode, start, end, bias, amount, exponent, start_offset, end_offset, fade, smooth = daemon_args

            # TODO? XYZ support for other channels
            if (i == 0) :
                mode = getattr(p, "DD_mode", mode)
                amount = getattr(p, "DD_amount", amount)
                start = getattr(p, "DD_start", start)
                end = getattr(p, "DD_end", end)
                bias = getattr(p, "DD_bias", bias)
                exponent = getattr(p, "DD_exponent", exponent)
                start_offset = getattr(p, "DD_start_offset", start_offset)
                end_offset = getattr(p, "DD_end_offset", end_offset)
                fade = getattr(p, "DD_fade", fade)
                smooth = getattr(p, "DD_smooth", smooth)

            if active:
                daemon_schedule_params = {
                    "start": start,
                    "end": end,
                    "bias": bias,
                    "amount": amount,
                    "exponent": exponent,
                    "start_offset": start_offset,
                    "end_offset": end_offset,
                    "fade": fade,
                    "smooth": smooth
                }
                
                self.daemon_data.append({
                    'name': f'Daemon {i+1}',
                    'mode': mode, 
                    'schedule': None, 
                    'schedule_params': daemon_schedule_params, 
                    'hires': hires, 
                    'multiplier': .1  # Add slider for this?
                })
                
                text = ",".join([
                    str(int(active)), str(int(hires)), mode, f"{amount}", f"{start}", f"{end}", f"{bias}", 
                    f"{exponent}", f"{start_offset}", f"{end_offset}", f"{fade:}", str(int(smooth))
                ])
                extra_gen_texts.append(f"D{i+1}:{text}")
        
        if extra_gen_texts:
            p.extra_generation_params['Detail Daemon'] = ";".join(extra_gen_texts)

        if not hasattr(self, 'callback_added'):
            on_cfg_denoiser(self.denoiser_callback)
            self.callback_added = True
        self.cfg_scale = p.cfg_scale
        self.batch_size = p.batch_size
        self.is_hires_pass = False 

    def before_process_batch(self, p, *args, **kwargs):
        self.is_hires_pass = False

    def before_hr(self, p, *args):
        self.is_hires_pass = True

    def postprocess(self, p, processed, *args):
        if hasattr(self, 'callback_added'):
            remove_callbacks_for_function(self.denoiser_callback)
            delattr(self, 'callback_added') 
        
    def denoiser_callback(self, params): 
        for daemon in self.daemon_data:
            if daemon['hires'] != self.is_hires_pass:
                continue

            name = daemon['name']
            mode = daemon['mode']            
            step = max(params.sampling_step, params.denoiser.step)
            steps = max(params.total_sampling_steps, params.denoiser.total_steps)
            actual_steps = steps - max(steps // params.denoiser.steps - 1, 0)
            idx = min(step, actual_steps - 1)

            if daemon['schedule'] is None:                
                daemon['schedule'] = self.make_schedule(actual_steps, **daemon['schedule_params'])
            
            schedule = daemon['schedule']   
            multiplier = schedule[idx] * daemon['multiplier']

            if is_forge:
                if idx == 0 and mode != "both":
                    tqdm.write(f'\033[33mDetail Daemon:\033[0m Forge does not support `cond` and `uncond` modes, using `both` instead')
                mode = "both"

            if mode == "cond":
                for i in range(self.batch_size):
                    params.sigma[i] *= 1 - multiplier
            elif mode == "uncond":
                for i in range(self.batch_size):
                    params.sigma[self.batch_size + i] *= 1 + multiplier
            else:
                params.sigma *= 1 - multiplier * self.cfg_scale

            if shared.opts.data.get("detail_daemon_verbose", False):
                tqdm.write(f'\033[32mDetail Daemon:\033[0m {name} | sigma: {params.sigma} | step: {idx}/{actual_steps - 1} | multiplier: {multiplier:.4f}')  
    
    def make_schedule(self, steps, start, end, bias, amount, exponent, start_offset, end_offset, fade, smooth):
        start = min(start, end)
        mid = start + bias * (end - start)
        multipliers = np.zeros(steps)

        start_idx, mid_idx, end_idx = [int(round(x * (steps - 1))) for x in [start, mid, end]]            

        start_values = np.linspace(0, 1, mid_idx - start_idx + 1)
        if smooth:  
            start_values = 0.5 * (1 - np.cos(start_values * np.pi))
        start_values = start_values ** exponent
        if start_values.any():
            start_values *= (amount - start_offset)  
            start_values += start_offset  

        end_values = np.linspace(1, 0, end_idx - mid_idx + 1)
        if smooth:
            end_values = 0.5 * (1 - np.cos(end_values * np.pi))
        end_values = end_values ** exponent
        if end_values.any():
            end_values *= (amount - end_offset)  
            end_values += end_offset  

        multipliers[start_idx:mid_idx+1] = start_values
        multipliers[mid_idx:end_idx+1] = end_values        
        multipliers[:start_idx] = start_offset
        multipliers[end_idx+1:] = end_offset    
        multipliers *= 1 - fade

        return multipliers

    def visualize(self, enabled, start, end, bias, amount, exponent, start_offset, end_offset, fade, smooth, mode, hires):
        try:
            steps = 50
            values = self.make_schedule(steps, start, end, bias, amount, exponent, start_offset, end_offset, fade, smooth)
            mean = sum(values)/steps
            peak = np.clip(max(abs(values)), -1, 1)
            if start > end:
                start = end
            mid = start + bias * (end - start)
            opacity = .1 + (1 - fade) * 0.7
            plot_color = (0.5, 0.5, 0.5, opacity) if not enabled else ((1 - peak)**2, 1, 0, opacity) if mean >= 0 else (1, (1 - peak)**2, 0, opacity) 

            plt.rcParams.update({
                "text.color":  plot_color, 
                "axes.labelcolor":  plot_color, 
                "axes.edgecolor":  plot_color, 
                "figure.facecolor":  (0.0, 0.0, 0.0, 0.0),  
                "axes.facecolor":    (0.0, 0.0, 0.0, 0.0),  
                "ytick.labelsize": 6,
                "ytick.labelcolor": plot_color,
                "ytick.color": plot_color,
            })       

            fig_main, ax_main = plt.subplots(figsize=(2.15, 2.00), layout="constrained")
            ax_main.plot(range(steps), values, color=plot_color, linewidth=1.5, linestyle="dashed" if hires else "solid")
            ax_main.axhline(y=0, color=plot_color, linestyle='dotted')
            ax_main.axvline(x=mid * (steps - 1), color=plot_color, linestyle='dotted')
            ax_main.tick_params(right=False, color=plot_color)
            ax_main.set_xticks([i * (steps - 1) / 10 for i in range(10)][1:])
            ax_main.set_xticklabels([])
            ax_main.set_ylim([-1, 1])
            ax_main.set_xlim([0, steps - 1])
            plt.close(fig_main)

            plot_color = (0.5, 0.5, 0.5, .1) if not enabled else (0.75, 0.75, 0.75, opacity) 
            plt.rcParams.update({
                "text.color":  plot_color, 
                "axes.labelcolor":  plot_color, 
                "axes.edgecolor":  plot_color, 
            }) 
            
            fig_thumb, ax_thumb = plt.subplots(figsize=(0.85, 0.85), layout="constrained")
            ax_thumb.plot(range(steps), values, color=plot_color, linewidth=1.5, linestyle="dashed" if hires else "solid")
            ax_thumb.set_xticks([])
            ax_thumb.set_yticks([])
            ax_thumb.set_ylim([-1, 1])
            ax_thumb.set_xlim([0, steps - 1])
            if (mode != "both"):
                ax_thumb.text(
                    0.98, 0.96, mode.upper(),  
                    transform=ax_thumb.transAxes,
                    fontsize=8, fontweight='bold', color=plot_color,
                    ha='right', va='top'
                )
            plt.close(fig_thumb)

            self.last_vis = fig_main
            self.last_thumb = fig_thumb
            return [fig_main, fig_thumb]
        except Exception:
            if self.last_vis is not None and self.last_thumb is not None:
                return [self.last_vis, self.last_thumb]
            return             

def xyz_support():
    for scriptDataTuple in scripts.scripts_data:
        if os.path.basename(scriptDataTuple.path) == 'xyz_grid.py':
            xy_grid = scriptDataTuple.module

            def confirm_mode(p, xs):
                for x in xs:
                    if x not in ['both', 'cond', 'uncond']:
                        raise RuntimeError(f'Invalid Detail Daemon Mode: {x}')
            mode = xy_grid.AxisOption(
                '[Detail Daemon] Mode',
                str,
                xy_grid.apply_field('DD_mode'),
                confirm=confirm_mode
            )
            amount = xy_grid.AxisOption(
                '[Detail Daemon] Amount',
                float,
                xy_grid.apply_field('DD_amount')
            )
            start = xy_grid.AxisOption(
                '[Detail Daemon] Start',
                float,
                xy_grid.apply_field('DD_start')
            )
            end = xy_grid.AxisOption(
                '[Detail Daemon] End',
                float,
                xy_grid.apply_field('DD_end')
            )
            bias = xy_grid.AxisOption(
                '[Detail Daemon] Bias',
                float,
                xy_grid.apply_field('DD_bias')
            )
            exponent = xy_grid.AxisOption(
                '[Detail Daemon] Exponent',
                float,
                xy_grid.apply_field('DD_exponent')
            )
            start_offset = xy_grid.AxisOption(
                '[Detail Daemon] Start Offset',
                float,
                xy_grid.apply_field('DD_start_offset')
            )
            end_offset = xy_grid.AxisOption(
                '[Detail Daemon] End Offset',
                float,
                xy_grid.apply_field('DD_end_offset')
            )
            fade = xy_grid.AxisOption(
                '[Detail Daemon] Fade',
                float,
                xy_grid.apply_field('DD_fade')
            )     
            smooth = xy_grid.AxisOption(
                '[Detail Daemon] Smooth',
                bool,
                xy_grid.apply_field('DD_smooth')
            )                                  
            xy_grid.axis_options.extend([
                mode,
                amount,
                start, 
                end, 
                bias, 
                exponent,
                start_offset,
                end_offset,
                fade,
                smooth,
            ])


try:
    xyz_support()
except Exception as e:
    tqdm.write(f'f"\033[31mDetail Daemon:\033[0m Error trying to add XYZ plot options for Detail Daemon: {e}')
