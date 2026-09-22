"""
批量分享命令
"""

from typing import List, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ..utils import get_client, handle_api_error, print_error, print_info, print_success, print_warning


def batch_share(
    output: Optional[str] = typer.Option(None, "--output", "-o", help="CSV输出文件名"),
    exclude: Optional[List[str]] = typer.Option(["来自：分享"], "--exclude", "-e", help="排除的目录名称模式"),
    dry_run: bool = typer.Option(False, "--dry-run", help="只扫描目录，不创建分享"),
    target_dir: Optional[str] = typer.Option(None, "--target-dir", "-t", help="指定起始目录路径（默认为根目录）"),
    depth: int = typer.Option(3, "--depth", "-d", help="扫描深度层级（默认3表示四级目录）"),
    share_level: str = typer.Option("folders", "--share-level", "-l", help="分享类型：folders/files/both（默认folders）")
):
    """
    批量分享目录/文件功能

    支持三种使用模式：
    1. 默认模式：分享三级目录下的所有文件夹（向后兼容）
    2. 指定目录模式：分享指定目录的子目录/文件
    3. 灵活深度模式：分享任意深度层级的目录/文件

    示例：
      quarkpan batch-share                                    # 默认行为
      quarkpan batch-share --target-dir "/我的资料"          # 指定目录
      quarkpan batch-share --depth 2 --share-level both     # 2级深度，文件+文件夹
    """
    console = Console()

    try:
        with get_client() as client:
            if not client.is_logged_in():
                print_error("未登录，请先使用 quarkpan auth login 登录")
                raise typer.Exit(1)

            from ...services.batch_share_service import BatchShareService

            batch_service = BatchShareService(client.api_client)

            # 显示参数信息
            if target_dir:
                print_info(f"🎯 指定目录模式: {target_dir}")
            print_info(f"📊 扫描深度: {depth} 级")
            print_info(f"📁 分享类型: {share_level}")

            # 显示排除模式
            if exclude:
                print_info(f"🚫 排除目录模式: {', '.join(exclude)}")

            # 收集目标目录
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("正在扫描目录...", total=None)
                target_directories = batch_service.collect_target_directories(
                    exclude_patterns=exclude,
                    target_dir=target_dir,
                    depth=depth,
                    share_level=share_level
                )
                progress.update(
                    task,
                    description=f"找到 {len(target_directories)} 个目标{'文件夹' if share_level == 'folders' else '项目' if share_level == 'both' else '文件'}")

            if not target_directories:
                print_warning("没有找到任何需要分享的目标项目")
                return

            # 显示目录结构预览
            item_type = "文件夹" if share_level == "folders" else ("文件" if share_level == "files" else "项目")
            print_success(f"找到 {len(target_directories)} 个目标{item_type}")

            # 创建目录预览表格
            table = Table(title=f"目标{item_type}预览")
            table.add_column("序号", style="cyan", width=4)
            table.add_column("类型", style="blue", width=6)
            table.add_column("名称", style="green", width=20)
            table.add_column("完整路径", style="dim", width=50)

            # 显示前20个目录作为预览
            preview_count = min(20, len(target_directories))
            for i, target_item in enumerate(target_directories[:preview_count], 1):
                item_type_icon = "📁" if target_item.get('is_folder', True) else "📄"
                item_name = target_item['name']
                full_path = target_item['full_path']

                table.add_row(
                    str(i),
                    item_type_icon,
                    item_name,
                    full_path
                )

            console.print(table)

            if len(target_directories) > preview_count:
                print_info(f"... 还有 {len(target_directories) - preview_count} 个目录")

            # 如果是dry run模式，只显示预览
            if dry_run:
                print_info("🔍 Dry run 模式：仅扫描，不创建分享")
                return

            # 确认是否继续
            if len(target_directories) > 10:
                confirm = typer.confirm(f"确定要为这 {len(target_directories)} 个目录创建分享链接吗？")
                if not confirm:
                    print_info("操作已取消")
                    return

            print_info("📤 开始批量创建分享链接...")

            # 批量创建分享
            with Progress(console=console) as progress:
                task = progress.add_task("创建分享链接...", total=len(target_directories))

                share_results = []
                for i, target_dir in enumerate(target_directories):
                    try:
                        progress.update(task, description=f"正在创建: {target_dir['name']}")

                        # 创建分享
                        share_result = client.shares.create_share(
                            file_ids=[target_dir['fid']],
                            title=target_dir['name'],
                            expire_days=0,  # 永久
                            password=None   # 无密码
                        )

                        if share_result:
                            from datetime import datetime
                            share_info = {
                                'target_directory': target_dir,
                                'share_result': share_result,
                                'share_title': target_dir['name'],
                                'share_url': share_result.get('share_url', ''),
                                'created_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'success': True
                            }
                            share_results.append(share_info)
                        else:
                            from datetime import datetime
                            share_info = {
                                'target_directory': target_dir,
                                'share_title': target_dir['name'],
                                'share_url': '',
                                'created_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'success': False,
                                'error': '分享创建失败'
                            }
                            share_results.append(share_info)

                    except Exception as e:
                        from datetime import datetime
                        share_info = {
                            'target_directory': target_dir,
                            'share_title': target_dir['name'],
                            'share_url': '',
                            'created_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'success': False,
                            'error': str(e)
                        }
                        share_results.append(share_info)

                    progress.advance(task)

            # 统计结果
            successful = sum(1 for result in share_results if result['success'])
            failed = len(share_results) - successful

            print_success(f"批量分享完成!")
            print_info(f"   总计: {len(share_results)} 个目录")
            print_info(f"   成功: {successful} 个")
            if failed > 0:
                print_warning(f"   失败: {failed} 个")

            # 导出CSV
            csv_path = batch_service.export_to_csv(share_results, output)
            print_success(f"📊 CSV文件已保存: {csv_path}")

            # 显示成功的分享链接（前10个）
            successful_shares = [r for r in share_results if r['success']]
            if successful_shares:
                print_info("📋 成功创建的分享链接:")
                result_table = Table()
                result_table.add_column("目录", style="cyan", width=20)
                result_table.add_column("分享链接", style="green", width=50)

                for result in successful_shares[:10]:  # 只显示前10个
                    result_table.add_row(
                        result['share_title'],
                        result['share_url']
                    )

                console.print(result_table)

                if len(successful_shares) > 10:
                    print_info(f"... 还有 {len(successful_shares) - 10} 个分享链接，详见CSV文件")

            # 显示失败的目录
            failed_shares = [r for r in share_results if not r['success']]
            if failed_shares:
                print_warning("❌ 创建失败的目录:")
                for result in failed_shares:
                    error_msg = result.get('error', '未知错误')
                    print_warning(f"   {result['share_title']}: {error_msg}")

    except Exception as e:
        handle_api_error(e, "批量分享")
        raise typer.Exit(1)


def list_structure(
    level: int = typer.Option(3, "--level", "-l", help="显示目录层级深度 (1-4)"),
    exclude: Optional[List[str]] = typer.Option(["来自：分享"], "--exclude", "-e", help="排除的目录名称模式")
):
    """查看网盘目录结构"""
    console = Console()

    try:
        with get_client() as client:
            if not client.is_logged_in():
                print_error("未登录，请先使用 quarkpan auth login 登录")
                raise typer.Exit(1)

            from ...services.batch_share_service import BatchShareService

            batch_service = BatchShareService(client.api_client)

            print_info(f"🔍 扫描网盘目录结构 (深度: {level} 级)...")

            if exclude:
                print_info(f"排除目录模式: {', '.join(exclude)}")

            # 显示目录结构
            if level >= 4:
                target_directories = batch_service.collect_target_directories(exclude)

                if target_directories:
                    # 按二级目录分组显示
                    from collections import defaultdict
                    grouped = defaultdict(lambda: defaultdict(list))

                    for target_dir in target_directories:
                        second = target_dir['second_level']
                        third = target_dir['third_level']
                        grouped[second][third].append(target_dir['name'])

                    print_success(f"找到 {len(target_directories)} 个目标目录")

                    for second_name, third_dirs in grouped.items():
                        print_info(f"\n📁 {second_name}/")
                        for third_name, targets in third_dirs.items():
                            print_info(f"  📁 {third_name}/")
                            for target_name in targets:
                                print_info(f"    📂 {target_name}")
                else:
                    print_warning("没有找到任何目标目录")
            else:
                print_info("此功能需要level=4来显示完整的四级目录结构")

    except Exception as e:
        handle_api_error(e, "查看目录结构")
        raise typer.Exit(1)
