import { AfterViewInit, Component, ElementRef, Input, OnChanges, OnDestroy, ViewChild } from '@angular/core';
import { Chart, registerables } from 'chart.js';

Chart.register(...registerables);

/** Thin wrapper: pass a Chart.js config, get a responsive chart. Re-renders when the config object changes. */
@Component({
  selector: 'app-chart',
  standalone: true,
  template: '<div class="chartbox"><canvas #c></canvas></div>'
})
export class ChartComponent implements AfterViewInit, OnChanges, OnDestroy {
  @Input() config: any;
  @ViewChild('c') canvas!: ElementRef<HTMLCanvasElement>;
  private chart?: Chart;

  ngAfterViewInit() { this.render(); }
  ngOnChanges() { if (this.canvas) this.render(); }
  ngOnDestroy() { this.chart?.destroy(); }

  private render() {
    this.chart?.destroy();
    if (!this.config) return;
    this.chart = new Chart(this.canvas.nativeElement, {
      ...this.config,
      options: { responsive: true, maintainAspectRatio: false, ...(this.config.options || {}) }
    });
  }
}
