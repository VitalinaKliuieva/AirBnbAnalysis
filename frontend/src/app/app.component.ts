import { Component, OnInit } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { forkJoin } from 'rxjs';
import { Api } from './api.service';
import { ChartComponent } from './chart.component';

const PAL = ['#ff5a5f', '#00a699', '#fc642d', '#484848', '#8a2be2', '#1e90ff', '#e6b800', '#2e8b57', '#c71585', '#767676'];
const METRICS = [
  { k: 'medianPrice', l: 'Median price / night' }, { k: 'medianPricePerGuest', l: 'Median price per guest' },
  { k: 'avgRating', l: 'Average rating (0-100)' }, { k: 'listings', l: 'Number of listings' },
  { k: 'totalReviews', l: 'Total reviews' }, { k: 'superhostPct', l: 'Superhosts (%)' },
  { k: 'entireHomePct', l: 'Entire homes (%)' }, { k: 'avgAccommodates', l: 'Avg guests accommodated' },
  { k: 'avgMinNights', l: 'Avg minimum nights' }, { k: 'instantPct', l: 'Instant bookable (%)' }
];

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [FormsModule, DecimalPipe, ChartComponent],
  template: `
  <header>
    <h1>Airbnb Market Explorer</h1>
    @if (meta) {
      <span class="badge" [class.demo]="meta.source === 'synthetic'">
        {{ meta.source === 'synthetic' ? 'DEMO DATA' : 'Real data' }} · {{ meta.listings | number }} listings · {{ meta.reviews | number }} reviews
      </span>
    }
  </header>
  <nav>
    @for (t of tabs; track t.k) { <button [class.on]="tab === t.k" (click)="tab = t.k">{{ t.l }}</button> }
  </nav>
  <main>
  @if (!meta) { <p>Loading… (the backend may still be reading the CSV files)</p> }
  @else {
    @if (meta.source === 'synthetic') {
      <p class="insight">Showing generated demo data. Put <b>Listings.csv</b> and <b>Reviews.csv</b> from the Maven Analytics Airbnb dataset into the <code>data/</code> folder and restart the backend.</p>
    }

    @if (tab === 'overview') {
      <section>
        <h2>City comparison</h2><div class="q">Can you spot any major differences in the Airbnb market between cities?</div>
        <div class="controls"><label>Metric
          <select [(ngModel)]="metric" (ngModelChange)="buildOverview()">
            @for (m of metrics; track m.k) { <option [value]="m.k">{{ m.l }}</option> }
          </select></label></div>
        <app-chart [config]="overviewChart" />
        <p class="insight">{{ overviewInsight }}</p>
        <app-chart [config]="mixChart" />
        <table>
          <tr><th>City</th><th>Listings</th><th>Median price</th><th>Per guest</th><th>Rating</th><th>Superhost %</th><th>Entire home %</th><th>Reviews</th></tr>
          @for (c of cities; track c.city) {
            <tr><td>{{ c.city }}</td><td>{{ c.listings | number }}</td><td>{{ c.medianPrice | number:'1.0-0' }}</td><td>{{ c.medianPricePerGuest | number:'1.0-0' }}</td>
                <td>{{ c.avgRating | number:'1.1-1' }}</td><td>{{ c.superhostPct | number:'1.0-0' }}</td><td>{{ c.entireHomePct | number:'1.0-0' }}</td><td>{{ c.totalReviews | number }}</td></tr>
          }
        </table>
      </section>
    }

    @if (tab === 'drivers') {
      <section>
        <h2>What drives price?</h2><div class="q">Which attributes have the biggest influence on price?</div>
        <div class="controls"><label>City
          <select [(ngModel)]="drvCity" (ngModelChange)="loadDrivers()"><option value="ALL">All cities</option>
            @for (c of meta.cities; track c) { <option [value]="c">{{ c }}</option> }</select></label>
          <span class="note">Red = numeric (|correlation| with ln price), teal = categorical (correlation ratio η). Click a bar for median price by group.</span></div>
        <app-chart [config]="driversChart" />
        <h2>Median price by: {{ groupFeature }}</h2>
        @if (groupsChart) { <app-chart [config]="groupsChart" /> } @else { <p class="note">Pick a categorical feature, Accommodates or Bedrooms.</p> }
        <p class="note">These are unadjusted associations, not causal effects; attributes are correlated with each other (e.g. guests and bedrooms).</p>
      </section>
    }

    @if (tab === 'reviews') {
      <section>
        <h2>Review trends & seasonality</h2><div class="q">Are you able to identify any trends or seasonality in the review data?</div>
        <div class="controls">
          @for (c of meta.cities; track c) {
            <label class="chip"><input type="checkbox" [checked]="selCities.has(c)" (change)="toggleCity(c)"> {{ c }}</label>
          }
        </div>
        <h2>Reviews per month</h2>
        <app-chart [config]="monthlyChart" />
        <h2>Seasonality index (1.0 = average month)</h2>
        <div class="controls"><label><input type="checkbox" [(ngModel)]="excl" (ngModelChange)="loadReviews()"> Exclude COVID years (2020–21)</label></div>
        <app-chart [config]="seasonChart" />
        <table><tr><th>City</th><th>Peak month</th><th>Quietest month</th></tr>
          @for (p of peaks; track p.city) { <tr><td>{{ p.city }}</td><td>{{ p.peak }}</td><td>{{ p.low }}</td></tr> }
        </table>
      </section>
    }

    @if (tab === 'value') {
      <section>
        <h2>Best value for travel</h2><div class="q">Which city offers a better value for travel?</div>
        <div class="controls">
          <label>Room type <select [(ngModel)]="valRoom" (ngModelChange)="loadValue()"><option value="ALL">All</option>
            @for (r of meta.roomTypes; track r) { <option [value]="r">{{ r }}</option> }</select></label>
          <label>Price ← {{ qw }}% quality weight → Quality
            <input type="range" min="0" max="100" step="5" [(ngModel)]="qw" (ngModelChange)="loadValue()"></label>
        </div>
        @if (valueRows.length) { <p class="insight"><b>{{ valueRows[0].city }}</b> scores best ({{ valueRows[0].score }}/100) at a {{ qw }}% quality / {{ 100 - qw }}% price weighting.</p> }
        <app-chart [config]="valueChart" />
        <table><tr><th>City</th><th>Score</th><th>Median price</th><th>Per guest</th><th>Rating (0-100)</th><th>Value rating (0-10)</th></tr>
          @for (v of valueRows; track v.city) {
            <tr><td>{{ v.city }}</td><td>{{ v.score | number:'1.0-1' }}</td><td>{{ v.medianPrice | number:'1.0-0' }}</td><td>{{ v.medianPricePerGuest | number:'1.0-0' }}</td>
                <td>{{ v.avgRating | number:'1.1-1' }}</td><td>{{ v.avgValueRating | number:'1.2-2' }}</td></tr>
          }
        </table>
        <p class="note">Score = w · quality (mean of rating and value-rating, relative to the best city) + (1−w) · affordability (cheapest median price per guest ÷ city's). Prices are in the dataset's currency, not converted.</p>
      </section>
    }
  }
  </main>`
})
export class AppComponent implements OnInit {
  tabs = [{ k: 'overview', l: 'City comparison' }, { k: 'drivers', l: 'Price drivers' }, { k: 'reviews', l: 'Reviews & seasonality' }, { k: 'value', l: 'Best value' }];
  tab = 'overview';
  metrics = METRICS; metric = 'medianPrice';
  meta: any; cities: any[] = []; roomMix: any[] = [];
  overviewChart: any; mixChart: any; overviewInsight = '';
  drvCity = 'ALL'; driversChart: any; groupsChart: any; groupFeature = '';
  selCities = new Set<string>(); excl = true; monthlyChart: any; seasonChart: any; peaks: any[] = [];
  qw = 50; valRoom = 'ALL'; valueChart: any; valueRows: any[] = [];

  constructor(private api: Api) {}

  ngOnInit() {
    this.api.get<any>('meta').subscribe(m => {
      this.meta = m;
      this.selCities = new Set(m.cities);
      forkJoin([this.api.get<any[]>('cities'), this.api.get<any[]>('cities/room-types')]).subscribe(([c, r]) => {
        this.cities = c; this.roomMix = r; this.buildOverview();
      });
      this.loadDrivers(); this.loadReviews(); this.loadValue();
    });
  }

  color(city: string) { return PAL[this.meta.cities.indexOf(city) % PAL.length]; }

  buildOverview() {
    const m = this.metrics.find(x => x.k === this.metric)!;
    const rows = [...this.cities].sort((a, b) => b[this.metric] - a[this.metric]);
    this.overviewChart = {
      type: 'bar',
      data: { labels: rows.map(r => r.city), datasets: [{ label: m.l, data: rows.map(r => r[this.metric]), backgroundColor: rows.map(r => this.color(r.city)) }] },
      options: { plugins: { legend: { display: false }, title: { display: true, text: m.l } } }
    };
    const hi = rows[0], lo = rows[rows.length - 1];
    this.overviewInsight = hi.city + ' is highest (' + hi[this.metric] + ') and ' + lo.city + ' is lowest (' + lo[this.metric] + ') on "' + m.l + '".';
    this.mixChart = {
      type: 'bar',
      data: {
        labels: this.meta.cities,
        datasets: this.meta.roomTypes.map((t: string, i: number) => ({
          label: t, backgroundColor: PAL[i % PAL.length],
          data: this.meta.cities.map((c: string) => { const r = this.roomMix.find(x => x.city === c && x.roomType === t); return r ? +(r.share * 100).toFixed(1) : 0; })
        }))
      },
      options: { plugins: { title: { display: true, text: 'Room type mix (% of listings)' } }, scales: { x: { stacked: true }, y: { stacked: true, max: 100 } } }
    };
  }

  loadDrivers() {
    this.api.get<any[]>('price/drivers', { city: this.drvCity }).subscribe(d => {
      this.driversChart = {
        type: 'bar',
        data: { labels: d.map(x => x.feature), datasets: [{ data: d.map(x => x.strength), backgroundColor: d.map(x => x.kind === 'numeric' ? '#ff5a5f' : '#00a699') }] },
        options: {
          indexAxis: 'y',
          plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c: any) => d[c.dataIndex].kind === 'numeric' ? 'r = ' + d[c.dataIndex].correlation : 'η = ' + d[c.dataIndex].strength } } },
          onClick: (_: any, els: any[]) => { if (els.length) this.loadGroups(d[els[0].index].feature); }
        }
      };
      this.loadGroups(this.groupFeature && d.some(x => x.feature === this.groupFeature) ? this.groupFeature : d[0].feature);
    });
  }

  loadGroups(feature: string) {
    this.groupFeature = feature;
    this.api.get<any[]>('price/by', { feature, city: this.drvCity }).subscribe(g => {
      this.groupsChart = g.length ? {
        type: 'bar',
        data: { labels: g.map(x => x.group), datasets: [{ label: 'Median price', data: g.map(x => x.medianPrice), backgroundColor: '#ff5a5f' }] },
        options: { plugins: { legend: { display: false }, tooltip: { callbacks: { afterLabel: (c: any) => g[c.dataIndex].count + ' listings' } } } }
      } : null;
    });
  }

  toggleCity(c: string) { this.selCities.has(c) ? this.selCities.delete(c) : this.selCities.add(c); this.loadReviews(); }

  loadReviews() {
    const cities = [...this.selCities].join(',');
    if (!cities) { this.monthlyChart = null; this.seasonChart = null; this.peaks = []; return; }
    this.api.get<any>('reviews/monthly', { cities }).subscribe(d => {
      this.monthlyChart = {
        type: 'line',
        data: { labels: d.months, datasets: Object.keys(d.series).map(c => ({ label: c, data: d.series[c], borderColor: this.color(c), borderWidth: 1.8, pointRadius: 0, tension: 0.2 })) },
        options: { interaction: { mode: 'index', intersect: false }, scales: { x: { ticks: { maxTicksLimit: 12 } } } }
      };
    });
    this.api.get<any>('reviews/seasonality', { cities, excludeCovid: this.excl }).subscribe(d => {
      this.seasonChart = {
        type: 'line',
        data: { labels: d.months, datasets: Object.keys(d.series).map(c => ({ label: c, data: d.series[c], borderColor: this.color(c), borderWidth: 2, pointRadius: 3, tension: 0.3 })) },
        options: { interaction: { mode: 'index', intersect: false } }
      };
      this.peaks = Object.keys(d.peak).map(c => ({ city: c, peak: d.peak[c], low: d.low[c] }));
    });
  }

  loadValue() {
    this.api.get<any[]>('value', { qualityWeight: this.qw / 100, roomType: this.valRoom }).subscribe(v => {
      this.valueRows = v;
      this.valueChart = {
        type: 'bar',
        data: { labels: v.map(x => x.city), datasets: [
          { label: 'Value score', data: v.map(x => x.score), backgroundColor: v.map(x => this.color(x.city)) }] },
        options: { indexAxis: 'y', plugins: { legend: { display: false }, tooltip: { callbacks: { afterLabel: (c: any) => 'quality ' + v[c.dataIndex].qualityIndex + ' · affordability ' + v[c.dataIndex].affordabilityIndex } } } }
      };
    });
  }
}
