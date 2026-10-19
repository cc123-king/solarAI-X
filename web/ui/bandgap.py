import gradio as gr
from api.bandgap import predict_perovskite_bandgap_wrapper, update_bandgap_inputs_visibility

def build_bandgap_tab():
    gr.Markdown("## 钙钛矿带隙预测")
    gr.Markdown("当前提供 MAPbIBr 原项目线性公式演示，结果需实验验证。CsMAFAPbIBr、MAFA、CsFA 缺少可核验的训练与验证资料，暂不可用。")
    with gr.Row():
        with gr.Column(scale=1):
            perovskite_type_bandgap = gr.Dropdown(
                choices=["MAPbIBr"],
                label="选择钙钛矿类型",
                value="MAPbIBr",
                interactive=True
            )
            with gr.Group(visible=True) as mapbi_group:
                Br_percentage_bg = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label="Br 百分比 (0-1)", interactive=True)
            with gr.Group(visible=False) as csmafa_group:
                gr.Markdown("请输入 Cs, FA, I 的比例 (请确保 Cs+MA+FA=1, I+Br=1，这里仅需输入 Cs, FA, I 比例)")
                Cs_ratio_bg = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.1, label="Cs 比例 (0-1)", interactive=True)
                FA_ratio_bg = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.8, label="FA 比例 (0-1)", interactive=True)
                I_ratio_bg = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.9, label="I 比例 (0-1)", interactive=True)
            with gr.Group(visible=False) as mafa_group:
                gr.Markdown("请输入 MA 和 I 的比例 (请确保 MA+FA=1, I+Br=1，这里仅需输入 MA, I 比例)")
                MA_ratio_bg = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label="MA 比例 (0-1)", interactive=True)
                I_ratio_mafa_bg = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.8, label="I 比例 (0-1)", interactive=True)
            with gr.Group(visible=False) as csfa_group:
                gr.Markdown("请输入 Cs 和 I 的比例 (请确保 Cs+FA=1, I+Br=1，这里仅需输入 Cs, I 比例)")
                Cs_ratio_csfa_bg = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.2, label="Cs 比例 (0-1)", interactive=True)
                I_ratio_csfa_bg = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.7, label="I 比例 (0-1)", interactive=True)
        with gr.Column(scale=1):
            predict_bandgap_btn = gr.Button("计算公式值", variant="primary")
            bandgap_result_text = gr.Textbox(label="预测结果", interactive=False)
            bandgap_status_text = gr.Markdown("_状态: 就绪_")
    perovskite_type_bandgap.change(
        fn=update_bandgap_inputs_visibility,
        inputs=[perovskite_type_bandgap],
        outputs=[mapbi_group, csmafa_group, mafa_group, csfa_group],
        queue=True
    )
    bandgap_method = gr.Dropdown(
        choices=[("原项目线性公式", "formula"), ("公开光谱文献基线", "measured_baseline")],
        value="formula", label="计算依据",
        info="文献基线显示独立留出误差，并限制为已有测量覆盖附近的组分；不提供新实验保证。",
    )
    all_bg_inputs = [
        perovskite_type_bandgap,
        Br_percentage_bg,
        Cs_ratio_bg, FA_ratio_bg, I_ratio_bg,
        MA_ratio_bg, I_ratio_mafa_bg,
        Cs_ratio_csfa_bg, I_ratio_csfa_bg, bandgap_method
    ]
    predict_bandgap_btn.click(
        fn=predict_perovskite_bandgap_wrapper,
        inputs=all_bg_inputs,
        outputs=[bandgap_result_text, bandgap_status_text],
        queue=True
    ) 