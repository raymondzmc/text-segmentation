// const model_types = ["BERT", "BERT-EXT"];

let width = (window.outerWidth - 30) / 2;
let height = window.outerHeight / 2;
let padding = 20;

let selected_token_idx = null
let attentions = null;
let importance = null;
let x_name = 'tsne_1';
let y_name = 'tsne_2';

let selection = {
    'Layer': null,
    'Head': null,
    'index': null,
    'type': 'Sub-word',
}
let cache = {
    'tokens': null
}



const server_query = d3.json('../api/projections', {
    method: "POST",
    // body: JSON.stringify({
    //     model_types: model_types
    // }),
    headers: {
        "Content-type": "application/json; charset=UTF-8"
    }
})


const hideLoader = () => $('#loader').css('display', 'none');
const showLoader = () => $('#loader').css('display', 'block');
const openModal = () => $('#modal').modal();

server_query.then(response => {

    const projectionSVG2 = d3.select("#projection_2")
        .append("ul")
        .attr("id", "tokens")
        .attr('class', 'token-container')
        .attr("width", width * 3)
        .attr("height", height);
        

    const projectionSVG1 = d3.select("#projection_1")
        .append("svg")
        .attr("class", "projection")
        .attr("width", width)
        .attr("height", height);

    const projectionSVG3 = d3.select("#projection_3")
        .append("svg")
        .attr("class", "projection")
        .attr('id', 'importanceSVG')
        .attr("width", width)
        .attr("height", height);

    // const projectionSVG4 = d3.select("#projection_4")
    //     .append("svg")
    //     .attr("class", "projection")
    //     .attr("width", width)
    //     .attr("height", height);
    renderProjections(response['projection'], projectionSVG1);
    console.log(response['importance'])
    renderImportance(response['importance'], projectionSVG3);
    hideLoader();
});


const parseProjectionData = (data) => {
    let domain = {
        xMin: 9999,
        xMax: -9999,
        yMin: 9999,
        yMax: -9999,
    }

    data.forEach(d => {
        d['ID'] = `${Math.round(d.ID)}`
        // d['label'] = Math.round(d.label)

        d[x_name] = +d[x_name];
        d[y_name] = +d[y_name];

        // Update the domain when necessary
        if (d[x_name] < domain.xMin) {
          domain.xMin = d[x_name]
        } else if (d[x_name] > domain.xMax) {
          domain.xMax = d[x_name]
        } 

        if (d[y_name] < domain.yMin) {
          domain.yMin = d[y_name]
        } else if (d[y_name] > domain.yMax) {
          domain.yMax = d[y_name]
        }
    });
    console.log(domain.xMin, domain.xMax);
    return data, domain
} 


const mouseover = d => {
    d3.selectAll(`.article.ID_${d.ID}`)
        .classed('selected', true)
        .attr("r", 8);
}

const mouseout = d => {
    d3.selectAll(`.article.ID_${d.ID}`)
        .classed('selected', false)
        .attr("r", 3);
}

const click = d => {
    // renderText(d.ID);
    // hashID = d.hashID;
    showLoader();
    const server_query = d3.json('../api/attn', {
        method: "POST",
        body: JSON.stringify({
            ID: d.ID
        }),
        headers: {
            "Content-type": "application/json; charset=UTF-8"
        }
    })

    server_query.then(response => {
        console.log(response);
        attentions = response['attentions'];
        // importance = response['head_importance'];
        renderModal(response['tokens'], '#tokens');
        projectionSVG3 = d3.select('#importanceSVG');
        // renderImportance(response['head_importance'], projectionSVG3);
        // renderModal(response['tokens'], '#tokens-right')
        cache['tokens'] = response['tokens'];
        hideLoader();
        // openModal();
    });
}

const renderText = (id) => {
    let svg = d3.select("#projection_2 > svg");
    let data = document_input[id]

    const yScale = d3
      .scaleBand()
      .domain(Array.from({length: data.length}, (_, i) => i + 1))
      .range([30, height - 20])
      .padding(0.01);

    let sentences = svg.selectAll('g').data(data);
    let sentencesEnter = sentences.enter().append('g').attr('class', 'row');

    sentences.merge(sentencesEnter)
        .attr('transform', (_, i) => `translate(0, ${yScale(i + 1)})`)
        .style('font-size', height / data.length);

    sentencesEnter.append('text')
        .merge(sentences.select('text'))
        .text(d => d);

    sentences.exit().remove()
    console.log(document_input[id]);
}

const renderColor = (color) => {
    selected_token_idx = selection.index;

    d3.selectAll('.clickable')
        .style('outline', 'none');

    d3.selectAll(`.token-${selected_token_idx}`)
        .style("outline", 'thin solid red');

    color.forEach((value, i) => {
        value = +value;
        if (value === 0) {
          bg_color = '#eee';
        } else {
          bg_color = d3.interpolateReds((value - 0) / 100);
        }
        d3.select(`#token-${i}`)
            .style('background-color', bg_color);
    })
}

const renderHeatmap = (color, gID, width, height, margin) => {
    let innerHeight = height - margin.top - margin.bottom;
    let innerWidth = width - margin.left - margin.right;

    let marginX = (width - height) / 2;
    let container = d3.select(gID);

    const xScale = d3
      .scaleBand()
      .domain(Array.from({length: color.length}, (_, i) => i + 1))
      .range([marginX, width - marginX])
      .padding(0.01);

    const yScale = d3
      .scaleBand()
      .domain(Array.from({length: color.length}, (_, i) => i + 1))
      .range([margin.top, margin.top + innerHeight])
      .padding(0.01);

    // Axis
    container.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0, ${yScale.range()[1]})`)
      .style('font-size', 15)
      .call(d3.axisBottom(xScale).tickSize(0))
      .select(".domain").remove()

    container.append('g')
      .attr('class', 'y-axis')
      .attr('transform', `translate(${marginX}, 0)`)
      .style('font-size', 15)
      .call(d3.axisLeft(yScale).tickSize(0))
      .select(".domain").remove()

    // Add classes to ticks
    d3.selectAll('.x-axis > .tick')
      .attr('class', d => `tick head-${d}`);
    d3.selectAll('.y-axis > .tick')
      .attr('class', d => `tick layer-${d}`);


    let rows = container
      .selectAll('.row')
      .data(color);

    let rowsEnter = rows.enter().append('g').attr('class', 'row');

    rowsEnter
      .merge(rows)
      .attr('transform', (d, i) => `translate(0, ${yScale(i + 1)})`)
      .attr('class', (d, i) => `layer-${i + 1}`)

    let grid = rowsEnter.merge(rows)
      .selectAll('.grid')
      .data(d => d);

    let gridEnter = grid.enter().append('g')
      .attr('class', 'grid')
      .attr('transform', (d, i) => `translate(${xScale(i + 1)}, 0)`);

    let rect = gridEnter.append('rect').merge(grid.selectAll('rect'));
    let rectEnter = rect.enter();

    rectEnter.merge(rect)
      .attr('class', (d, i) => `head-${i + 1}`)
      .attr('height', yScale.bandwidth())
      .attr('width', xScale.bandwidth())
      .attr('ry', yScale.bandwidth() * 0.1)
      .attr('rx', xScale.bandwidth() * 0.1)
      .attr('fill', d => d3.interpolateReds(d))
      .on('click', function(event) {
        let clicked = d3.select(this);
        let head = clicked.attr('class').match(/(\d+)/)[0];
        let layer = d3.select(this.parentNode.parentNode).attr('class').match(/(\d+)/)[0];
        selection.Layer = layer;
        selection.Head = head;

        // Unselect previous selections 
        d3.selectAll('.grid > .selected').classed('selected', false);
        d3.selectAll('.x-axis > .tick.selected').classed('selected', false);
        d3.selectAll('.y-axis > .tick.selected').classed('selected', false);

        // Select heatmap cell and tick on axis
        clicked.classed('selected', true);
        d3.select(`.x-axis > .tick.head-${head}`).classed('selected', true);
        d3.select(`.y-axis > .tick.layer-${layer}`).classed('selected', true);

        if (selection.index !== null) {
          renderColor(attentions[layer - 1][head - 1][selected_token_idx]);
        }
      })

    gridEnter.append('text').merge(grid.selectAll('text'))
      // .attr("text-anchor", "middle")
      .attr('dx', `${xScale.bandwidth() / 4}`)
      .attr('dy', `${yScale.bandwidth() / 2}`)
      .style('font-size', `${yScale.bandwidth() / 3}px`)
      .text(d => d.toFixed(2));

}


const renderModal = (tokens, container) => {
    

    // containerName = container.split('-')[1];

    li = d3.select(container).selectAll("li").data(tokens);

    li_enter = li.enter().append("li");

    li_enter.merge(li)
        .transition(300)
        .text(d => d)
        .attr('id', (d, i) => `token-${i}`)
        .attr('class', (d, i) => `clickable token-${i}`)
        .style('background-color', '#eee');


    let width = $("#modal").width();
    let height = $("#modal").height() / 2;
    let margin = {'top': 20, 'left': 0, 'right': 0, 'bottom': 50}

    d3.select('svg.svg-container').remove();

    $(function(){
        $(".clickable").on("click", function() {
            console.log("clicked")
            console.log(selection.Layer, selection.Head)
            if (selection.Layer !== null && selection.Head !== null){
              selected_token_idx = +$(this).attr('id').split('-')[1];
              selection.index = selected_token_idx;
              attn = attentions[selection.Layer - 1][selection.Head - 1];
              color = attn[selected_token_idx];
              console.log(color);
              renderColor(color);
            }
        });
    });

    li.exit().remove();

}

const renderProjections = (data, svg) => {
    
    data, domain = parseProjectionData(data);

    const xScale = d3.scaleLinear().domain([domain.xMin, domain.xMax]).range([padding, width - padding])
    const yScale = d3.scaleLinear().domain([domain.yMax, domain.yMin]).range([padding, height - padding])

    console.log(data, domain);
    let samples = svg.selectAll(".article").data(data);
    let samplesEnter = samples.enter().append("circle");
    samplesEnter
        .attr("class", d => `article ID_${Math.round(d.ID)}`)
        .attr("r", 3)
        .attr("cx", d => xScale(d[x_name]))
        .attr("cy", d => yScale(d[y_name]))
        .attr('fill', d => d3.interpolateRdYlBu(1 - d.confidence)); // Confidence

    samplesEnter
        .on('mouseover', mouseover)
        .on("mouseout", mouseout)
        .on("click", click);
}

const renderImportance = (data, svg) => {
    marginX = 20;
    marginY = 20;
    const xScale = d3
      .scaleBand()
      .domain(Array.from({length: data.length}, (_, i) => i + 1))
      .range([marginX, width - marginX])
      .padding(0.01);

    const yScale = d3
      .scaleBand()
      .domain(Array.from({length: data.length}, (_, i) => i + 1))
      .range([marginY, height - marginY])
      .padding(0.01);

    // Axis
    svg.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0, ${yScale.range()[1]})`)
      .style('font-size', 15)
      .call(d3.axisBottom(xScale).tickSize(0))
      .select(".domain").remove()

    svg.append('g')
      .attr('class', 'y-axis')
      .attr('transform', `translate(${marginX}, 0)`)
      .style('font-size', 15)
      .call(d3.axisLeft(yScale).tickSize(0))
      .select(".domain").remove()

    let attn_image = d3.select('#projection_4 > svg').append('svg:image');

    // Add classes to ticks
    d3.selectAll('.x-axis > .tick')
      .attr('class', d => `tick head-${d}`);
    d3.selectAll('.y-axis > .tick')
      .attr('class', d => `tick layer-${d}`);

    let rows = svg
      .selectAll('.row')
      .data(data);

    let rowsEnter = rows.enter().append('g').attr('class', 'importance-row');

    rowsEnter
      .merge(rows)
      .attr('transform', (d, i) => `translate(0, ${yScale(i + 1)})`)
      .attr('class', (d, i) => `layer-${i + 1}`)

    let grid = rowsEnter.merge(rows)
      .selectAll('.grid')
      .data(d => d);

    let gridEnter = grid.enter().append('g')
      .attr('class', 'grid')
      .attr('transform', (d, i) => `translate(${xScale(i + 1)}, 0)`);

    let rect = gridEnter.append('rect').merge(grid.selectAll('rect'));
    let rectEnter = rect.enter();

    rectEnter.merge(rect)
      .attr('class', (d, i) => `head-${i + 1}`)
      .attr('height', yScale.bandwidth())
      .attr('width', xScale.bandwidth())
      .attr('ry', yScale.bandwidth() * 0.1)
      .attr('rx', xScale.bandwidth() * 0.1)
      .attr('fill', d => d3.interpolateReds(d))
      .on('click', function(event) {
        let clicked = d3.select(this);
        let head = clicked.attr('class').match(/(\d+)/)[0];
        let layer = d3.select(this.parentNode.parentNode).attr('class').match(/(\d+)/)[0];
        selection.Layer = layer;
        selection.Head = head;

        // Unselect previous selections 
        d3.selectAll('.grid > .selected').classed('selected', false);
        d3.selectAll('.x-axis > .tick.selected').classed('selected', false);
        d3.selectAll('.y-axis > .tick.selected').classed('selected', false);

        // Select heatmap cell and tick on axis
        clicked.classed('selected', true);
        d3.select(`.x-axis > .tick.head-${head}`).classed('selected', true);
        d3.select(`.y-axis > .tick.layer-${layer}`).classed('selected', true);

        renderColor(attentions[selection.Layer - 1][selection.Head - 1][selected_token_idx]);
      })

    gridEnter.append('text').merge(grid.selectAll('text'))
      // .attr("text-anchor", "middle")
      .attr('dx', `${xScale.bandwidth() / 4}`)
      .attr('dy', `${yScale.bandwidth() / 2}`)
      .style('font-size', `${yScale.bandwidth() / 3}px`)
      .text(d => d.toFixed(2));
    // data, domain = parseProjectionData(data);

    // const xScale = d3.scaleLinear().domain([domain.xMin, domain.xMax]).range([padding, width - padding]);
    // const yScale = d3.scaleLinear().domain([domain.yMax, domain.yMin]).range([padding, height - padding]);

}