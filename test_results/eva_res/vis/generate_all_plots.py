"""
主可视化生成脚本
统一生成所有Nature系列期刊适用的可视化图表
"""

import sys
import os
from pathlib import Path
import time

# 添加各个模块的路径
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, 'config'))
sys.path.append(os.path.join(current_dir, 'data_processing'))
sys.path.append(os.path.join(current_dir, 'visualization_scripts'))

from visualization_scripts.performance_matrix import PerformanceMatrixVisualizer
from visualization_scripts.radar_charts import TopModelRadarVisualizer
from visualization_scripts.forest_plots import ELOForestPlotVisualizer
from visualization_scripts.dimension_elo_radar import DimensionELORadarVisualizer
from visualization_scripts.score_violin_and_forest import ScoreViolinAndForestVisualizer
from data_processing.data_processor import DataProcessor


class MasterVisualizationGenerator:
    """主可视化生成器"""
    
    def __init__(self):
        self.output_dir = Path(__file__).parent / "plots"
        print(f"可视化输出目录: {self.output_dir.absolute()}")
        
    def generate_all_visualizations(self, formats=['png', 'pdf']):
        """
        生成所有可视化图表
        """
        start_time = time.time()
        
        print("="*60)
        print("🎨 Nature系列期刊可视化图表生成器")
        print("="*60)
        print()
        
        total_plots = 0
        
        try:
            # 明确打印将使用的 ELO 数据文件
            try:
                _dp = DataProcessor()
                _dp.load_elo_scores()
                if getattr(_dp, 'last_elo_path', None):
                    print(f"📄 可视化将使用的 ELO 数据文件: {_dp.last_elo_path}")
            except Exception as e:
                print(f"❌ 预加载 ELO 数据失败: {e}")
                raise

            # 1. 性能矩阵热图
            print("📊 1. 生成性能矩阵热图...")
            print("-" * 40)
            matrix_viz = PerformanceMatrixVisualizer()
            matrix_viz.create_heatmap(save_format=formats)
            matrix_viz.create_annotated_heatmap(save_format=formats)
            matrix_viz.create_clustered_heatmap(save_format=formats)
            total_plots += 3
            print("✅ 性能矩阵热图生成完成 (3个图表)")
            print()
            
            # 2. 顶级模型雷达图
            print("🎯 2. 生成顶级模型雷达图...")
            print("-" * 40)
            radar_viz = TopModelRadarVisualizer()
            radar_viz.create_single_radar_chart(save_format=formats)
            radar_viz.create_individual_radar_charts(save_format=formats)
            radar_viz.create_comparative_radar_chart(save_format=formats)
            radar_viz.create_selected_models_radar(save_format=formats)
            radar_viz.create_selected_models_radar_relative(save_format=formats)
            radar_viz.create_selected_models_elo_radar(save_format=formats)
            radar_viz.create_selected_models_elo_radar_relative(save_format=formats)
            total_plots += 7
            print("✅ 顶级模型雷达图生成完成 (7个图表)")
            print()
            
            # 3. ELO森林图
            print("🌲 3. 生成ELO森林图...")
            print("-" * 40)
            forest_viz = ELOForestPlotVisualizer()
            forest_viz.create_overall_elo_forest_plot(save_format=formats)
            forest_viz.create_tiered_elo_forest_plot(save_format=formats)
            forest_viz.create_elo_uncertainty_analysis(save_format=formats)
            # 聚类ELO森林图（基于四维ELO，Ward聚类；无scipy时自动回退）
            print("3.b 生成聚类ELO森林图（Ward聚类）...")
            forest_viz.create_clustered_elo_forest_plot(save_format=formats)
            total_plots += 4
            print("✅ ELO森林图生成完成 (4个图表，包括聚类版)")
            print()
            
            # 4. 分维度ELO雷达图
            print("📡 4. 生成分维度ELO雷达图...")
            print("-" * 40)
            dim_elo_viz = DimensionELORadarVisualizer()
            dim_elo_viz.create_all_models_elo_radar(save_format=formats)
            dim_elo_viz.create_top_models_elo_radar(save_format=formats)
            dim_elo_viz.create_comparative_elo_analysis(save_format=formats)
            dim_elo_viz.create_dimension_ranking_comparison(save_format=formats)
            total_plots += 4
            print("✅ 分维度ELO雷达图生成完成 (4个图表)")
            print()

            # 5. 评分小提琴图与评分森林图（非ELO）
            print("🎻 5. 生成评分小提琴图与评分森林图...")
            print("-" * 40)
            score_viz = ScoreViolinAndForestVisualizer()
            score_viz.create_violin_plot(save_format=formats)
            score_viz.create_score_forest(save_format=formats, orientation='vertical')
            score_viz.create_score_forest(save_format=formats, orientation='horizontal')
            total_plots += 3
            print("✅ 评分小提琴图与评分森林图生成完成 (3个图表)")
            print()
            
            # 生成完成
            elapsed_time = time.time() - start_time
            print("="*60)
            print("🎉 所有可视化图表生成完成！")
            print(f"📈 总计生成: {total_plots} 个图表")
            print(f"⏱️  总耗时: {elapsed_time:.2f} 秒")
            print(f"💾 输出格式: {', '.join(formats)}")
            print("="*60)
            print()
            
            # 显示文件结构
            self.show_output_structure()
            
        except Exception as e:
            print(f"❌ 生成过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
    
    def show_output_structure(self):
        """显示输出文件结构"""
        print("📁 输出文件结构:")
        print("-" * 40)
        
        def print_tree(directory, prefix="", max_depth=3, current_depth=0):
            if current_depth > max_depth:
                return
            
            items = sorted(directory.iterdir()) if directory.exists() else []
            dirs = [item for item in items if item.is_dir()]
            files = [item for item in items if item.is_file()]
            
            for i, item in enumerate(dirs + files):
                is_last = i == len(dirs + files) - 1
                current_prefix = "└── " if is_last else "├── "
                print(f"{prefix}{current_prefix}{item.name}")
                
                if item.is_dir() and current_depth < max_depth:
                    extension_prefix = "    " if is_last else "│   "
                    print_tree(item, prefix + extension_prefix, max_depth, current_depth + 1)
        
        print_tree(self.output_dir)
        print()

    def generate_main_figures(self, formats=['png', 'pdf']):
        """生成主图（四幅）：
        a) 分数森林图（LLM-Judge，纵向、zoom 4–9）
        b) ELO 森林图（纵向）
        c) 选定模型分数雷达图（中心3分）
        d) 选定模型ELO相对雷达图（中心40%）
        并将文件统一导出到 plots/main_figures 下。
        """
        main_dir = self.output_dir / 'main_figures'
        main_dir.mkdir(parents=True, exist_ok=True)

        # a) 分数森林图（纵向、zoom 版）
        score_viz = ScoreViolinAndForestVisualizer()
        fig, _ = score_viz.create_score_forest(orientation='vertical', save_format=[])  # 我们自定义另存
        # 采用zoom 4–9视图
        ax = fig.axes[0]
        ax.set_ylim(4, 9)
        for fmt in formats:
            path = main_dir / f'main_a_score_forest.{fmt}'
            fig.savefig(path, dpi=300, bbox_inches='tight')
        plt_close_safe(fig)

        # b) ELO 聚类森林图（固定K=4，主图版本），标注 mu [lower, upper]
        elo_viz = ELOForestPlotVisualizer()
        fig, _ = elo_viz.create_clustered_elo_forest_plot(save_format=[], n_clusters=4)
        for fmt in formats:
            path = main_dir / f'main_b_elo_forest_cluster_k4.{fmt}'
            fig.savefig(path, dpi=300, bbox_inches='tight')
        plt_close_safe(fig)

        # b1) 自动聚类版（SI 可选）
        fig, _ = elo_viz.create_clustered_elo_forest_plot(save_format=[], n_clusters=None)
        for fmt in formats:
            path = main_dir / f'main_b1_elo_forest_cluster_auto.{fmt}'
            fig.savefig(path, dpi=300, bbox_inches='tight')
        plt_close_safe(fig)

        # c) 分数雷达（3–10）
        radar_viz = TopModelRadarVisualizer()
        fig, _ = radar_viz.create_selected_models_radar(save_format=[])
        for fmt in formats:
            path = main_dir / f'main_c_score_radar.{fmt}'
            fig.savefig(path, dpi=300, bbox_inches='tight')
        plt_close_safe(fig)

        # d) ELO 相对雷达（40–100%）
        fig, _ = radar_viz.create_selected_models_elo_radar_relative(save_format=[])
        for fmt in formats:
            path = main_dir / f'main_d_elo_radar_relative.{fmt}'
            fig.savefig(path, dpi=300, bbox_inches='tight')
        plt_close_safe(fig)

        print(f"✅ 主图已导出到: {main_dir}")

    def generate_supplement_figures(self, formats=['png', 'pdf']):
        """生成补充图（带 Kendall τ 的相关性图）到 plots/supplement_figures。"""
        supp_dir = self.output_dir / 'supplement_figures'
        supp_dir.mkdir(parents=True, exist_ok=True)

        dim_viz = DimensionELORadarVisualizer()
        # 正文版（Pearson+Spearman）
        fig, _ = dim_viz.create_comparative_elo_analysis(save_format=[])
        for fmt in formats:
            path = supp_dir / f'sup_perf_elo_corr_pearson_spearman.{fmt}'
            fig.savefig(path, dpi=300, bbox_inches='tight')
        plt_close_safe(fig)

        # 补充版（再加 Kendall τ）
        fig, _ = dim_viz.create_comparative_elo_analysis_with_kendall(save_format=[])
        for fmt in formats:
            path = supp_dir / f'sup_perf_elo_corr_with_kendall.{fmt}'
            fig.savefig(path, dpi=300, bbox_inches='tight')
        plt_close_safe(fig)

        print(f"✅ 补充图已导出到: {supp_dir}")


def plt_close_safe(fig):
    try:
        import matplotlib.pyplot as _plt
        _plt.close(fig)
    except Exception:
        pass

class MasterVisualizationGenerator(MasterVisualizationGenerator):
    def generate_summary_report(self):
        """生成图表摘要报告"""
        report_content = """# 可视化图表生成报告

## 图表类型和用途

### 1. 性能矩阵热图 (Performance Matrix)
- **基础热图**: 显示9个模型在4个维度的性能分数
- **详细注释热图**: 包含统计信息的增强版热图
- **聚类热图**: 展示模型和维度相似性的层次聚类

### 2. 顶级模型雷达图 (Top Model Radar Charts)  
- **综合雷达图**: 顶级3个模型的多维度比较
- **独立雷达图**: 每个顶级模型的单独详细分析
- **比较雷达图**: 绝对性能vs相对性能对比

### 3. ELO森林图 (ELO Forest Plots)
- **整体ELO森林图**: 所有模型的TrueSkill ELO排名
- **分层森林图**: 按性能层级分组的ELO展示
- **不确定性分析图**: ELO评分的置信区间分析

### 4. 分维度ELO雷达图 (Dimension ELO Radar Charts)
- **所有模型ELO雷达图**: 基于ELO评分的多模型比较
- **顶级模型ELO雷达图**: 重点关注前3名模型
- **相关性分析图**: 性能分数vs ELO评分的相关性
- **排名比较热图**: 各维度排名的可视化矩阵

## Nature期刊适配特点

1. **高分辨率输出**: 所有图表支持300 DPI的PNG、PDF、SVG格式
2. **专业配色**: 参考期刊标准的颜色方案和字体
3. **统计严谨性**: 包含置信区间、p值、相关系数等统计指标
4. **清晰标注**: 详细的图例、轴标签和数值标注

## 数据来源

- **性能评分**: 豆包Seed-1.6评分数据 (4维度)
- **ELO评分**: TrueSkill算法计算的贝叶斯技能评估
- **模型范围**: 9个主流AI模型 (排除专门化学模型)
- **评估维度**: 正确性、完备性、理论深度、论述严谨性与信息密度

## 使用建议

1. **论文主图**: 推荐使用顶级模型雷达图和ELO森林图
2. **补充材料**: 性能矩阵和详细分析图表
3. **方法学贡献**: 突出LLM评分vs人工评分的一致性分析
4. **结果展示**: 重点展示MOSES、O3、GPT-4.1的性能优势

生成时间: {timestamp}
"""
        
        report_path = self.output_dir / "visualization_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content.format(timestamp=time.strftime("%Y-%m-%d %H:%M:%S")))
        
        print(f"📋 图表摘要报告已保存: {report_path}")


def main():
    """主函数"""
    generator = MasterVisualizationGenerator()
    
    # 生成所有可视化
    generator.generate_all_visualizations(formats=['png', 'pdf'])
    # 生成主图（四幅）
    try:
        generator.generate_main_figures(formats=['png', 'pdf'])
    except Exception as e:
        print(f"❌ 主图导出失败: {e}")
        import traceback; traceback.print_exc()
    # 生成补充图（相关性：Pearson+Spearman+Kendall）
    try:
        generator.generate_supplement_figures(formats=['png', 'pdf'])
    except Exception as e:
        print(f"❌ 补充图导出失败: {e}")
        import traceback; traceback.print_exc()
    
    # 生成摘要报告
    generator.generate_summary_report()


if __name__ == "__main__":
    main()
